"""Join the two halves of one perf-swe32 job locally: DeltaAI server evidence (verified by verify_evidence.py) <->
tbench run (the pools' trajectories, timelines, close receipts, host samples and the SWE-bench evaluation), one row per
instance x server, plus per-server throughput, accuracy and failures.

Usage:
  python -B join_swe32.py <check dir> <tbench run dir> [--tokenizer-python PY] [--out DIR]

<check dir>: verify_evidence.py's output (local-verification.json; its `evidence_dir` holds the extracted files).
<tbench run dir>: runs/<run_id> from the fetched package: <label>/{instances.txt, timeline.jsonl, closes/, runs/<id>/
<id>/<id>.traj.json, runs/<id>/preds.json, preds.json, host-samples.jsonl, spec.json, model_override.yaml, eval/}.

Per runtime server and session (= the instance's index in the server's task list), pilot-v14's join (join_pilot.py)
for the session pool: the agent's model calls (assistant messages and format errors) against the session's runtime
rows (calls = rows, response ids in order, usage = receipts: prompt tokens N, completion C' - N + 1), every format error
followed by a rebuild, api_calls, preds.json against the submission; the close receipt the pool saved
(closes/<id>.json) equal to the session's close row in sessions.jsonl; the timeline entry; with --tokenizer-python the
exact-token re-render of every request through the sealed release's render path (a fresh clone of the sealed bundle):
the agent <-> server proof. The stock control has no server evidence: its calls and usage only.

Throughput (config.throughput, PROVISIONAL): main-agent completion tokens
of every call of every instance (client usage) / span, span = first runner start to last runner end on the tbench clock
(timeline.jsonl, close included); also the span from the first model request to the last response (trajectory
timestamps). Cross-checks of the numerator: the server's generated tokens (C' - N + 1 per row) and vLLM's
generation_tokens_total delta over the sampler window (it also counts requests cancelled at a stop). 2026-11-10
(perf review FINDINGS 2.8): an instance cut at the stop without a trajectory kept its runner time and lost its tokens;
`throughput_all` adds its session's server-generated tokens (measurements.jsonl; None for the stock control, which has
no server rows), `throughput_runner` stays client tokens only (the eval pipeline checks it against 'Gen tok (client)' /
'Span s'). The batch profile (clean cells, decode time lost) is the job's own, or recomputed from the samples for a job
built before 2026-11-10 (decode_profile, `batch_source`). Accuracy: resolved /
planned from the evaluation (an instance without a prediction is an empty patch); the task list is the sealed one
(pins tasks.provisional says so per run).
Exit 0 (JOIN_PASS) or 1 (JOIN_FAIL); EVALUATION_ABSENT is reported, not failed.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
if os.name == 'nt':  # local only (2026-11-11): the tokenizer / git children without a console window of their own
    sys.path.insert(0, str(CANDIDATE))
    import localtemp
    localtemp.hide_windows()
MODEL_PINS = {'config.json': ('model-config.json', '69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c'),
              'tokenizer.json': ('tokenizer.json', '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'),
              'tokenizer_config.json': ('tokenizer-config.json', '5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b')}
COLUMNS = [('instance_id', 'Instance'), ('label', 'Server'), ('policy', 'Policy'), ('status', 'Session'),
           ('exit_status', 'Exit status'), ('resolved', 'Resolved'), ('eval_status', 'Eval'), ('fail_to_pass', 'F2P'),
           ('pass_to_pass', 'P2P'), ('calls', 'Calls'), ('rows', 'Requests'), ('format_errors', 'Format errors'),
           ('rebuilds', 'Rebuilds'), ('completion_tokens', 'Completion tok'), ('max_prompt_tokens', 'Max prompt tok'),
           ('wall_s', 'Wall s'), ('close_http', 'Close'), ('patch_chars', 'Patch chars'), ('join', 'Join'),
           ('exact', 'Exact tokens')]
SERVER_COLUMNS = [('label', 'Server'), ('policy', 'Policy'), ('instances', 'Instances'), ('done', 'Done'),
                  ('resolved', 'Resolved'), ('accuracy', 'Accuracy %'), ('completion_tokens', 'Completion tok (client)'),
                  ('server_generated_tokens', 'Generated tok (server)'), ('vllm_generated_tokens', 'Generated tok (vLLM)'),
                  ('span_runner_s', 'Span s (runner)'), ('throughput_runner', 'tok/s (runner span)'),
                  ('no_trajectory', 'Started without trajectory'),
                  ('throughput_all', 'tok/s (runner span; server tokens for instances without a trajectory)'),
                  ('span_model_s', 'Span s (model calls)'), ('throughput_model', 'tok/s (model span)'),
                  ('decode_lost_pct', 'Decode time lost % of ITL sum'),
                  ('calls', 'Calls'), ('turns_per_min', 'Turns/min'), ('context_refusals', 'Context refusals'),
                  ('timeouts', 'Timeouts'), ('capacity_waits', 'CAPACITY_WAIT'), ('joined', 'Joined'),
                  ('kernels', 'Kernels (attention/FA/GDN prefill/sampler)'),
                  # 2026-11-12 (goal_query servers; empty elsewhere): the Goal wait on the sessions' critical path
                  ('goal_wait_share', 'Goal wait share of session wall'), ('goal_wait_s', 'Goal wait s (sum)'),
                  ('goal_wait_turns', 'Goal-wait turns'), ('goal_wait_ms_p50', 'Goal wait ms p50 (waiting turns)'),
                  ('goal_wait_ms_p95', 'Goal wait ms p95 (waiting turns)'),
                  # 2026-11-12 (--release-idle-pages servers; empty elsewhere): capacity.jsonl's totals
                  ('idle_evictions', 'Idle evictions'), ('idle_readmissions', 'Readmissions'),
                  ('idle_readmitted_pages', 'Readmitted Pages'),
                  # 2026-11-16: the steady state (in-flight sessions == K; steady_state()) next to the whole run and
                  # the tail, and the thinking-on servers' rollbacks / rebuilds (keep-rebuild-1116)
                  ('steady_s', 'Steady s (in flight = K)'), ('steady_tok_s', 'tok/s (steady, K in flight)'),
                  ('steady_turns_per_min', 'Turns/min (steady)'), ('tail_s', 'Tail s'),
                  ('tail_tok_s', 'tok/s (tail)'), ('tail_turns_per_min', 'Turns/min (tail)'),
                  ('rollbacks', 'Rollbacks'), ('thinking_rebuilds', 'Rebuilds (thinking history)')]
MB = 1e6
STEADY_NOTE = ('Steady state (2026-11-16): the runner timeline (timeline.jsonl, tbench clock) gives the in-flight '
               'sessions at every moment (an instance is in flight from its runner start to its runner end); the steady '
               'window is every stretch in which exactly K are in flight (K = the server\'s max_running_sessions; then '
               'at least K tasks remain unfinished by construction), from the first moment the pool fills to the last '
               'moment it is still full; the tail is the rest of the runner span after it. tok/s = client completion '
               'tokens of the calls whose response time (the trajectory timestamp) falls in the window / its length; '
               'turns/min = those calls / its minutes. The whole-run columns are unchanged (runner span).')


def idle_release(out):
    """2026-11-12: a --release-idle-pages server's evictions and readmissions (capacity.jsonl rows), {} elsewhere."""
    path = out / 'capacity.jsonl'
    if not path.is_file():
        return {}
    events = load_jsonl(path)
    return dict(idle_evictions=sum(1 for e in events if e.get('event') == 'evict'),
                idle_readmissions=sum(1 for e in events if e.get('event') == 'readmit'),
                idle_readmitted_pages=sum(e.get('pages') or 0 for e in events if e.get('event') == 'readmit'))


def goal_wait(server, records, out):
    """2026-11-12 (the 64-instance proxy: how much of the run time goal construction takes, JCT-relevant): the time
    the sessions' requests wait for their Goal on the critical path, as a share of the sessions' wall time. Per
    completed request of a goal_query server, measurements.jsonl `goal_async_ms` (the runtime's native goal_async_ms:
    the admission parked from its Goal's start until it resumed; with --goal-overlap only the part of the Goal that phase
    1 did not hide, 0.0 when the Goal was ready at the join) is time the client's request could not progress. Returns
    goal_wait_s = the sum over the completed requests, goal_wait_share = goal_wait_s / the sum of the instances' runner
    walls (Wall s: runner start to end on tbench, the instance's JCT), goal_wait_turns = requests that waited (> 0),
    their p50 / p95 in ms, and goal_wait_unknown = requests without the field. Requests that failed or were cut at the
    stop have no row and are not counted. {} for every other server. goal-rollback-1116: a rollback whose Goal carried
    (measurements goal_carried) builds no Goal and waits for none: counted apart (goal_wait_carried), not unknown."""
    if server['kind'] != 'runtime' or server['policy'] != 'goal_query' or not (out / 'measurements.jsonl').is_file():
        return {}
    measured = load_jsonl(out / 'measurements.jsonl')
    carried = sum(1 for m in measured if m.get('goal_carried') is True)
    waits = [m.get('goal_async_ms') for m in measured if m.get('goal_carried') is not True]
    known = [float(w) for w in waits if type(w) in (int, float)]
    waited = sorted(w for w in known if w > 0)
    walls = sum(r['wall_s'] for r in records if type(r.get('wall_s')) in (int, float))
    total = sum(known) / 1000.0
    pick = lambda q: round(waited[int(round(q * (len(waited) - 1)))], 1) if waited else None  # noqa: E731
    return dict(goal_wait_s=round(total, 1), goal_wait_share=round(total / walls, 4) if walls > 0 else None,
                goal_wait_turns=len(waited), goal_wait_ms_p50=pick(0.5), goal_wait_ms_p95=pick(0.95),
                goal_wait_unknown=len(waits) - len(known), goal_wait_wall_s=round(walls, 1),
                **({'goal_wait_carried': carried} if carried else {}))


def load(path):
    return json.loads(Path(path).read_bytes())


def load_jsonl(path):
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def native_path(out, request_id):
    return Path(out) / ('native-' + hashlib.sha256(request_id.encode()).hexdigest() + '.json')


def agent_calls(trajectory):
    """Model calls of a Mini-SWE-Agent 2.4.2 trajectory in order: assistant messages and format-error messages
    (join_pilot.agent_calls), with the call's timestamp."""
    messages = trajectory['messages']
    calls = []
    for index, message in enumerate(messages):
        extra = message.get('extra') or {}
        if message.get('role') == 'assistant':
            kind, reply = 'assistant', message.get('content')
        elif message.get('role') == 'user' and extra.get('interrupt_type') == 'FormatError':
            kind, reply = 'format_error', extra.get('model_response')
        else:
            continue
        response = extra.get('response') or {}
        calls.append(dict(index=index, kind=kind, response_id=response.get('id'), usage=response.get('usage') or {},
                          reply=reply, timestamp=extra.get('timestamp'), created=response.get('created'),
                          request=[dict(role=m['role'], content=m['content'],
                                        # thinking on: the reasoning the agent sent back (reasoning_content)
                                        **({'reasoning': m['reasoning_content']} if m.get('role') == 'assistant'
                                           and isinstance(m.get('reasoning_content'), str) else {}))
                                   for m in messages[:index] if m.get('role') in ('system', 'user', 'assistant')]))
    return calls


EDGE_CLIENT = 'edge_script'  # controller.EDGE_CLIENT (2026-11-16: plan keepedge's scripted client)
# The runtime's reasoning fields of a client's assistant message, in order (thinking.CLIENT_FIELDS at 7a8c36e, read by
# thinking.client_reasoning from the request's wire body; the transcripts' key is `reasoning`)
CLIENT_REASONING_FIELDS = ('reasoning_content', 'reasoning')


def wire_messages(messages):
    """A request body's messages as the runtime's transcripts receive them (mswea_route.attach_reasoning ->
    thinking.attach): role and content; an assistant message's reasoning from the first string field of
    CLIENT_REASONING_FIELDS as `reasoning` (none: no key); every other key dropped (vLLM's request model keeps role and
    content). 2026-11-17: keepedgei r2 (job 3613280) rendered the scripted requests raw and every session's call 2 was
    refused by mswea_protocol.validate_messages (an assistant message with `reasoning_content`)."""
    out = []
    for m in messages:
        item = dict(role=m.get('role'), content=m.get('content'))
        if m.get('role') == 'assistant':
            value = next((m[f] for f in CLIENT_REASONING_FIELDS if isinstance(m.get(f), str)), None)
            if value is not None:
                item['reasoning'] = value
        out.append(item)
    return out
EDGE_REQUIRED = ('first_reply_dropped', 'double_rollback', 'resend')  # every scripted session exercises these


def script_record(trajectory):
    """The scripted client's record (tbench/agent_ext/swe32_edge_script.py: the handover message's
    extra.swe32_script), or None."""
    found = [m['extra']['swe32_script'] for m in trajectory.get('messages') or ()
             if isinstance(m.get('extra'), dict) and isinstance(m['extra'].get('swe32_script'), dict)]
    return found[0] if len(found) == 1 else None


def script_calls(record):
    """The scripted session's calls in agent_calls' shape: kind format_error for a dropped reply (the next call is a
    rollback), resend for the request sent again, else assistant; with each call's expected history."""
    calls = record.get('calls') or []
    out = []
    for k, call in enumerate(calls):
        following = calls[k + 1]['expected_history'] if k + 1 < len(calls) else None
        kind = 'format_error' if following == 'rollback' else 'resend' if call.get('action') == 'resend-first' \
            else 'assistant'
        response = call.get('response') or {}
        out.append(dict(index=k, kind=kind, response_id=response.get('id'), usage=response.get('usage') or {},
                        reply=(call.get('reply') or {}).get('content'), timestamp=call.get('timestamp'),
                        created=response.get('created'), request=wire_messages(call.get('request') or []),
                        expected_history=call.get('expected_history'), action=call.get('action')))
    return out


def session_rows(out, session):
    rows = sorted((r for r in load_jsonl(Path(out) / 'runtime.jsonl') if r.get('session_index') == session),
                  key=lambda r: r['turn'])
    return rows, [load(native_path(out, r['request_id'])) for r in rows]


def generated_of(rows, natives, k):
    if k + 1 >= len(rows) or rows[k + 1]['stage_b']['rebuild'] or not rows[k + 1]['stage_b']['local_turn']:
        return None
    N = len(natives[k]['prompt_token_ids'])
    g = rows[k]['stage_b']['computed_end'] - N + 1
    return natives[k + 1]['prompt_token_ids'][N:N + g]


OK_STATUSES = ('PASS', 'PENDING')  # PENDING: a server-row check that needs swe32-bulk.tar.xz (fetched later)


def join_instance(evidence, pool, server, session, instance, summary, task, close_rows, have_bulk=True):
    """One instance of one server: trajectory <-> server rows (runtime; PENDING without the bulk archive), close
    receipt, timeline, preds."""
    label = server['label']
    record = dict(instance_id=instance, label=label, policy=server['policy'], session=session, checks=[], calls=0, rows=0)
    add = lambda name, ok, detail='': record['checks'].append(dict(check=name, status='PASS' if ok else 'FAIL',
                                                                    detail=str(detail)[:2000]))
    outcome = (summary.get('instances') or {}).get(instance) or {}
    record['status'] = outcome.get('status')
    timeline = {r['instance_id']: r for r in load_jsonl(pool / 'timeline.jsonl')}
    entry = timeline.get(instance) or {}
    record.update(start=entry.get('start'), end=entry.get('end'), close_http=entry.get('close_http'),
                  wall_s=round(entry['end'] - entry['start'], 3) if entry.get('start') and entry.get('end') else None)
    path = pool / 'runs' / instance / instance / (instance + '.traj.json')
    runtime = server['kind'] == 'runtime'
    rows, natives = session_rows(evidence / 'job-output' / label, session) if runtime and have_bulk else ([], [])
    record['rows'] = len(rows)
    record['rebuilds'] = sum(1 for r in rows if r['stage_b']['rebuild'])
    if not path.is_file():
        done = outcome.get('status') == 'done'
        add('trajectory-present', not done, 'no trajectory' + (' although the job recorded it done' if done else ''))
        record['join'] = 'NO_TRAJECTORY'
        return record, None
    trajectory = load(path)
    info = trajectory.get('info') or {}
    calls = agent_calls(trajectory)
    scripted = server.get('client') == EDGE_CLIENT  # 2026-11-16: plan keepedge's scripted client
    script = script_record(trajectory) if scripted else None
    if scripted:
        calls = script_calls(script or {})
    record.update(calls=len(calls), format_errors=sum(1 for c in calls if c['kind'] == 'format_error'),
                  exit_status=info.get('exit_status'), api_calls=(info.get('model_stats') or {}).get('api_calls'),
                  patch_chars=len(info.get('submission') or ''),
                  completion_tokens=sum((c['usage'] or {}).get('completion_tokens') or 0 for c in calls),
                  prompt_tokens_total=sum((c['usage'] or {}).get('prompt_tokens') or 0 for c in calls),
                  max_prompt_tokens=max([(c['usage'] or {}).get('prompt_tokens') or 0 for c in calls] or [0]),
                  call_tokens=[(c['timestamp'], (c['usage'] or {}).get('completion_tokens') or 0) for c in calls
                               if isinstance(c['timestamp'], (int, float))],
                  first_call_ts=min([c['timestamp'] for c in calls if isinstance(c['timestamp'], (int, float))], default=None),
                  last_call_ts=max([c['timestamp'] for c in calls if isinstance(c['timestamp'], (int, float))], default=None))
    messages = trajectory.get('messages') or []
    content = messages[1].get('content') if len(messages) > 1 and isinstance(messages[1], dict) else None
    add('task-message-is-the-sealed-statement', isinstance(content, str) and task['problem_statement'] in content,
        'messages[1] must carry the sealed problem statement of ' + instance)
    if scripted:  # one agent query ran the whole scripted session (the handover), which ended without an error
        coverage = (script or {}).get('coverage') or {}
        record.update(script_coverage=coverage, script_error=(script or {}).get('error'))
        add('api-calls', record['api_calls'] == 1 and len(calls) > 0, '%s api_calls (the handover), %d scripted calls'
            % (record['api_calls'], len(calls)))
        add('edge-script-complete', script is not None and script.get('error') is None
            and all(coverage.get(k) == 1 for k in EDGE_REQUIRED),
            dict(error=(script or {}).get('error'), coverage=coverage))
    else:
        add('api-calls', record['api_calls'] == len(calls), '%s api_calls, %d calls' % (record['api_calls'], len(calls)))
    add('timeline-entry', bool(entry), 'timeline.jsonl names the instance')
    preds_path = pool / 'runs' / instance / 'preds.json'
    if preds_path.is_file():
        pred = load(preds_path).get(instance) or {}
        add('preds-equal-submission', (pred.get('model_patch') or '') == (info.get('submission') or ''), '')
    else:
        add('preds-present', False, 'runs/%s/preds.json missing' % instance)
    add('job-instance-record', outcome.get('exit_status') == info.get('exit_status')
        and outcome.get('api_calls') == record['api_calls'] and outcome.get('patch_chars') == record['patch_chars'],
        dict(job={k: outcome.get(k) for k in ('status', 'exit_status', 'api_calls', 'patch_chars')}))
    spec = None
    if runtime and not have_bulk:
        record['checks'].append(dict(check='server-rows', status='PENDING', detail='swe32-bulk.tar.xz not fetched yet: '
                                     'calls = rows, usage = receipts, response order, close receipt and the exact '
                                     're-render are checked when fetch_run.ps1 runs again without -CoreOnly'))
    elif runtime:
        ids, row_ids = [c['response_id'] for c in calls], [r['response_id'] for r in rows]
        if record['status'] == 'truncated':
            # a runner killed at the stop: its last reply may have been served (a row) but never saved
            add('calls-equal-rows-truncated', 0 <= len(rows) - len(calls) <= 1,
                '%d calls, %d rows (truncated: at most one unsaved reply)' % (len(calls), len(rows)))
        else:
            add('calls-equal-rows', len(calls) == len(rows), '%d calls, %d rows' % (len(calls), len(rows)))
        position = {rid: i for i, rid in enumerate(row_ids)}
        matched = [position.get(i) for i in ids]
        add('response-ids-in-order', None not in matched and matched == sorted(matched),
            dict(unmatched_calls=[k for k, m in enumerate(matched) if m is None][:10]))
        usage = []
        for k, m in enumerate(matched):
            if m is None:
                continue
            N = len(natives[m]['prompt_token_ids']); g = rows[m]['stage_b']['computed_end'] - N + 1
            got = calls[k]['usage']
            if got.get('prompt_tokens') != N or got.get('completion_tokens') != g:
                usage.append(dict(call=k, usage=got, server=dict(prompt_tokens=N, completion_tokens=g)))
        add('usage-equals-receipts', not usage, usage[:5])
        unrebuilt = [k for k, c in enumerate(calls) if c['kind'] == 'format_error' and matched[k] is not None
                     and matched[k] + 1 < len(rows) and not rows[matched[k] + 1]['stage_b']['rebuild']]
        add('format-errors-rebuild-the-server-session', not unrebuilt, dict(calls=unrebuilt))
        if scripted:  # every row's thinking history is the one the script expects (keep-rebuild-1116)
            wrong = [dict(call=k, expected=c['expected_history'], row=(rows[m].get('thinking') or {}).get('history'))
                     for k, (c, m) in enumerate(zip(calls, matched)) if m is not None
                     and (rows[m].get('thinking') or {}).get('history') != c['expected_history']]
            add('edge-history-labels', not wrong and None not in matched, wrong[:10])
            record['rollbacks_after_capacity_wait'] = sum(
                1 for m in matched if m is not None and (rows[m].get('thinking') or {}).get('history') == 'rollback'
                and (rows[m].get('capacity_waits') or 0) > 0)
        close_path = pool / 'closes' / (instance + '.json')
        row = close_rows.get(session)
        if close_path.is_file() and record['close_http'] == 200 and row is not None:
            receipt = load(close_path)
            add('close-receipt-equals-sessions-row', receipt.get('object') == 'session.close'
                and {k: v for k, v in receipt.items() if k not in ('object', 'repeated')} == row,
                dict(object=receipt.get('object'), repeated=receipt.get('repeated')))
        elif row is not None:
            add('close-receipt-present', record['status'] == 'truncated' or (summary.get('closed_by_job') and any(
                c['session'] == session for c in summary['closed_by_job'])), 'closed by the job (stopped pool)')
        if None not in matched and len(set(matched)) == len(matched):
            spec = [dict(messages=c['request'], reply=c['reply'] or '', prompt=natives[m]['prompt_token_ids'],
                         stable_end=natives[m]['stable_end'], span=natives[m]['observation_span'],
                         rebuild=rows[m]['stage_b']['rebuild'], generated=generated_of(rows, natives, m))
                    for c, m in zip(calls, matched)]
    record['join'] = 'PASS' if all(c['status'] in OK_STATUSES for c in record['checks']) else 'FAIL'
    return record, spec


def evaluation(pool, records, planned):
    """Attach the pool's evaluation (eval/eval-summary.json) to its records; returns the evaluation's checks."""
    checks = []
    add = lambda name, ok, detail='': checks.append(dict(check=name, status='PASS' if ok else 'FAIL',
                                                         detail=str(detail)[:2000]))
    summary_path = pool / 'eval' / 'eval-summary.json'
    if not summary_path.is_file():
        return checks, 'absent'
    value = load(summary_path)
    predictions = {r['instance_id']: r for r in load_jsonl(pool / 'eval' / 'predictions.jsonl')}
    add('eval-predictions-cover-the-plan', sorted(predictions) == sorted(planned), dict(missing=sorted(set(planned) -
                                                                                                set(predictions))))
    preds = load(pool / 'preds.json') if (pool / 'preds.json').is_file() else {}
    differing = [i for i, r in predictions.items() if (r.get('model_patch') or '') != ((preds.get(i) or {}).get('model_patch') or '')]
    add('eval-predictions-equal-preds', not differing, dict(differing=differing))
    add('eval-report-present', not value.get('report_missing'), value.get('report'))
    unevaluated = []
    for record in records:
        item = (value.get('instances') or {}).get(record['instance_id']) or {}
        record.update(eval_status=item.get('status'), resolved=item.get('resolved'),
                      fail_to_pass=item.get('fail_to_pass'), pass_to_pass=item.get('pass_to_pass'))
        if record.get('patch_chars') and item.get('status') in ('empty_patch', None):
            unevaluated.append(record['instance_id'])
    add('eval-ran-every-patch', not unevaluated, unevaluated)
    return checks, 'present'


def clone_release(target):
    sys.path.insert(0, str(CANDIDATE))
    import git_bundle
    pins = load(CANDIDATE / 'pins.json')
    repo = Path(target) / 'repo'
    git_bundle.git(['-c', 'core.autocrlf=false', 'clone', '-q', '--branch', 'sealed-release',
                    CANDIDATE / git_bundle.BUNDLE_NAME, repo], stderr=subprocess.STDOUT)
    head = git_bundle.git(['-C', repo, 'rev-parse', 'HEAD']).decode().strip()
    if head != pins['commit']:
        raise ValueError('sealed bundle clone is not the pinned commit: ' + head)
    return repo


def render_worker(spec_path, result_path):
    """Runs under the tokenizer Python: re-render every call of every session with the release's XML render path
    (join_pilot.render_worker, unchanged in substance)."""
    spec = load(spec_path)
    work = Path(tempfile.mkdtemp(prefix='swe32-join-'))
    results = {}
    try:
        repo = clone_release(work)
        overlay = repo / 'source-overlay'
        evidence = overlay / 'remote_ra_linux' / 'model-evidence'
        model = work / 'model'; model.mkdir()
        for name, (source, pin) in MODEL_PINS.items():
            raw = (evidence / source).read_bytes()
            if hashlib.sha256(raw).hexdigest() != pin:
                raise ValueError('release model evidence differs from the pin: ' + source)
            (model / name).write_bytes(raw)
        sys.path.append(str(overlay))
        from transformers import AutoTokenizer
        from runtime_renderer import EngineSourceRenderer
        tok = AutoTokenizer.from_pretrained(str(model), local_files_only=True)

        def render_template(template, conversation, kwargs):
            return tok.apply_chat_template(conversation, chat_template=template, tokenize=False, return_dict=False, **kwargs)
        make = lambda: EngineSourceRenderer(evidence_dir=evidence, tokenizer=tok, render_template=render_template,
                                            verify_identity=lambda: True, page_tokens=spec.get('page_tokens', 16))
        thinking = bool(spec.get('thinking'))  # 2026-11-06: a thinking-on server's transcripts (the XML keep path)
        for key, calls in spec['sessions'].items():
            transcript = make().transcript(**(dict(thinking=True) if thinking else {}))
            mismatch, rendered = None, 0
            for k, call in enumerate(calls):
                try:
                    tokens, stable, span = transcript.render(call['messages'])
                except Exception as exc:
                    mismatch = dict(call=k, error='%s: %s' % (type(exc).__name__, exc))
                    break
                rebuild = transcript.pending['rebuild']
                got = (list(tokens), stable, list(span) if span else None, rebuild)
                want = (call['prompt'], call['stable_end'], call['span'], call['rebuild'])
                if got != want:
                    first = next((i for i, (a, b) in enumerate(zip(got[0], want[0])) if a != b), min(len(got[0]), len(want[0])))
                    mismatch = dict(call=k, first_token_difference=first, rendered_tokens=len(got[0]),
                                    server_tokens=len(want[0]), stable=(got[1], want[1]), span=(got[2], want[2]),
                                    rebuild=(got[3], want[3]))
                    break
                rendered += 1
                generated = call['generated']
                if generated is None:
                    generated = tok.encode(call['reply'], add_special_tokens=False)
                transcript.commit(tokens, generated)
            results[key] = dict(calls=len(calls), rendered=rendered, exact=mismatch is None and rendered == len(calls),
                                mismatch=mismatch)
        value = dict(status='DONE', commit=spec['commit'], sessions=results)
        if spec.get('keep_render'):  # 2026-11-16 (release gate 1): the sealed tool, the pinned template
            try:
                import stock_history
                value['keep_render'] = keep_render_sessions(
                    repo / KEEP_TOOL, lambda text: tok.encode(text, add_special_tokens=False),
                    lambda template, messages, kwargs: tok.apply_chat_template(messages, chat_template=template,
                                                                               tokenize=False, **kwargs),
                    stock_history.pinned_template(model),
                    lambda ids: tok.decode(list(ids), skip_special_tokens=False),
                    {key: spec['sessions'][key] for key in spec['keep_render']})
            except Exception as exc:
                import traceback
                value['keep_render'] = dict(status='ERROR', error='%s: %s' % (type(exc).__name__, exc),
                                            traceback=traceback.format_exc())
    except Exception as exc:
        import traceback
        value = dict(status='ERROR', error='%s: %s' % (type(exc).__name__, exc), traceback=traceback.format_exc())
    finally:
        def writable(function, path, error):
            os.chmod(path, stat.S_IWRITE); function(path)
        shutil.rmtree(work, onexc=writable)
    with Path(result_path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=1)
        stream.write('\n')
    return value


def server_timeline(out):
    """The server's batch timeline (measure.batch_timeline: every sample checked against its receipt), or None."""
    sys.path.insert(0, str(CANDIDATE))
    import measure
    try:
        return measure.batch_timeline(out)
    except Exception:
        return None


def vllm_generated(out, timeline=False):
    """vLLM's generation_tokens_total over the sampler window (last readable sample - first), or None."""
    timeline = server_timeline(out) if timeline is False else timeline
    if timeline is None:
        return None
    values = [p['generated'] for p in timeline if p['ok'] and p['generated'] is not None]
    return round(values[-1] - values[0]) if len(values) >= 2 else None


def decode_profile(recorded, timeline):
    """(batch profile, source): the job's own when it has the 2026-11-10 columns (clean cells, decode_lost); else,
    for a job built before them, the profile recomputed from its samples by this harness's measure.py, used only if it
    reproduces every existing column of the recorded one (postrun/itl_reverify.py); else the recorded one."""
    if recorded is not None and 'decode_lost' in recorded:
        return recorded, 'job'
    if timeline is None:
        return recorded, 'recorded (no samples to recompute)'
    sys.path.insert(0, str(HERE))
    import measure
    import itl_reverify
    profile = measure.batch_profile(timeline)
    if recorded is not None and itl_reverify.existing_differ(recorded, profile):
        return recorded, 'recorded (the recomputation differs in the existing columns)'
    return profile, 'recomputed from the samples (measure.py of this harness)'


def all_tokens(server, records, out, completion):
    """(tokens, note): the client completion tokens of every instance with a trajectory plus, for an instance that
    started but left no trajectory (cut at the stop, FINDINGS 2.8), the server's generated tokens of its session
    (measurements.jsonl gen_tokens, C' - N + 1 per request); None where those are unknown (the stock control has no
    server rows). Equal to `completion` when every started instance has a trajectory."""
    missing = [r for r in records if r.get('join') == 'NO_TRAJECTORY' and r.get('start')]
    if not missing:
        return completion, 'client'
    if server['kind'] != 'runtime' or not (out / 'measurements.jsonl').is_file():
        return None, '%d instance(s) without a trajectory, no server tokens' % len(missing)
    per_session = {}
    for m in load_jsonl(out / 'measurements.jsonl'):
        per_session[m.get('session')] = per_session.get(m.get('session'), 0) + (m.get('gen_tokens') or 0)
    added = sum(per_session.get(r['session'], 0) for r in missing)
    return completion + added, 'client + server tokens of %d instance(s) without a trajectory (%d tokens)' % (
        len(missing), added)


def markdown(rows, columns):
    lines = ['| ' + ' | '.join(title for _, title in columns) + ' |', '|' + ' --- |' * len(columns)]
    lines += ['| ' + ' | '.join('' if row.get(key) is None else str(row.get(key)) for key, _ in columns) + ' |'
              for row in rows]
    return '\n'.join(lines)


def steady_state(records, calls, K):
    """(steady, tail) of one server (STEADY_NOTE): records' runner start / end, calls [(timestamp, completion
    tokens)], K. Each part: dict(seconds, tokens, calls, tok_s, turns_per_min, start, end); None when the pool never
    held K in flight."""
    timed = [r for r in records if r.get('start') is not None and r.get('end') is not None]
    events = sorted([(r['start'], 1) for r in timed] + [(r['end'], -1) for r in timed], key=lambda e: (e[0], e[1]))
    if not events or not K:
        return None, None
    windows, running, opened = [], 0, None
    for t, step in events:
        running += step
        if running == K and opened is None:
            opened = t
        elif running != K and opened is not None:
            if t > opened:
                windows.append((opened, t))
            opened = None
    if not windows:
        return None, None
    end_all = max(t for t, _ in events)

    def part(spans):
        seconds = sum(b - a for a, b in spans)
        inside = [(ts, tok) for ts, tok in calls if any(a <= ts < b for a, b in spans)]
        tokens = sum(tok for _, tok in inside)
        return dict(seconds=round(seconds, 3), tokens=tokens, calls=len(inside), start=round(spans[0][0], 3),
                    end=round(spans[-1][1], 3), tok_s=round(tokens / seconds, 2) if seconds > 0 else None,
                    turns_per_min=round(len(inside) / seconds * 60, 2) if seconds > 0 else None)
    steady = part(windows)
    tail = part([(windows[-1][1], end_all)]) if end_all > windows[-1][1] else None
    return steady, tail


def server_row(server, records, summary, evidence, pool, eval_state, running=None):
    label = server['label']
    done = [r for r in records if r.get('status') == 'done']
    evaluated = [r for r in records if r.get('eval_status') is not None]
    starts = [r['start'] for r in records if r.get('start')]
    ends = [r['end'] for r in records if r.get('end')]
    firsts = [r['first_call_ts'] for r in records if r.get('first_call_ts')]
    lasts = [r['last_call_ts'] for r in records if r.get('last_call_ts')]
    completion = sum(r.get('completion_tokens') or 0 for r in records)
    span_runner = round(max(ends) - min(starts), 3) if starts and ends else None
    span_model = round(max(lasts) - min(firsts), 3) if firsts and lasts else None
    measured = summary.get('measurements') or {}
    out = evidence / 'job-output' / label
    failures = load_jsonl(out / 'failures.jsonl')
    timeline = server_timeline(out)
    tokens_all, tokens_all_note = all_tokens(server, records, out, completion)
    batch, batch_source = decode_profile(measured.get('batch'), timeline)
    lost = (batch or {}).get('decode_lost') or {}
    pct = lambda key: round(100 * lost[key], 1) if lost.get(key) is not None else None
    row = dict(label=label, policy=server['policy'], instances=len(records), done=len(done),
               resolved=sum(1 for r in evaluated if r.get('resolved')) if eval_state == 'present' else None,
               completion_tokens=completion, server_generated_tokens=measured.get('gen_tokens'),
               vllm_generated_tokens=vllm_generated(out, timeline), span_runner_s=span_runner, span_model_s=span_model,
               throughput_runner=round(completion / span_runner, 2) if span_runner else None,
               # 2026-11-10 (FINDINGS 2.8): with a cut instance's tokens (all_tokens); = throughput_runner without one
               tokens_all=tokens_all, tokens_all_note=tokens_all_note,
               throughput_all=round(tokens_all / span_runner, 2) if span_runner and tokens_all is not None else None,
               no_trajectory=sum(1 for r in records if r.get('join') == 'NO_TRAJECTORY' and r.get('start')),
               decode_lost_pct=pct('lost_share'), decode_lost_pct_low=pct('lost_share_low'),
               decode_lost_pct_high=pct('lost_share_high'), batch_source=batch_source,
               throughput_model=round(completion / span_model, 2) if span_model else None,
               calls=sum(r.get('calls') or 0 for r in records),
               turns_per_min=round(sum(r.get('calls') or 0 for r in records) / span_runner * 60, 2) if span_runner else None,
               context_refusals=sum(1 for f in failures if 'maximum context length' in json.dumps(f).lower()),
               timeouts=sum(1 for r in records if 'timeout' in str(r.get('exit_status') or '').lower()),
               capacity_waits=measured.get('capacity_waits'), joined=sum(1 for r in records if r.get('join') == 'PASS'),
               evaluation=eval_state, prompt_tokens=measured.get('prompt_tokens'),
               prompt_buckets=measured.get('prompt_buckets'), admission_stall_fraction=measured.get('admission_stall_fraction'),
               queue_wait_s=measured.get('queue_wait_s'), selective_turns=measured.get('selective_turns'),
               rebuilds=measured.get('rebuilds'), goal=measured.get('goal'), batch=batch)
    row['accuracy'] = round(100.0 * row['resolved'] / row['instances'], 2) if row['resolved'] is not None else None
    if server.get('client') == EDGE_CLIENT:  # 2026-11-16: the scripted edge-case smoke's coverage
        covered = {k: sum((r.get('script_coverage') or {}).get(k) or 0 for r in records) for k in (
            'first_reply_dropped', 'rollback_after_closed', 'double_rollback', 'rollback_after_cut', 'cut_attempts',
            'resend')}
        covered['rollback_during_capacity_wait'] = sum(r.get('rollbacks_after_capacity_wait') or 0 for r in records)
        row.update(edge_coverage=covered, edge_failures=len(failures),
                   edge_not_exercised=sorted(k for k in ('rollback_after_cut', 'rollback_during_capacity_wait')
                                             if not covered[k]))
    row.update(goal_wait(server, records, out))  # 2026-11-12: goal_query servers only
    row.update(idle_release(out))  # 2026-11-12: --release-idle-pages servers only
    # 2026-11-16: the steady state (K in flight) and the tail (STEADY_NOTE)
    steady, tail = steady_state(records, [c for r in records for c in r.get('call_tokens') or ()], running)
    row.update(running=running, steady=steady, tail=tail,
               steady_s=(steady or {}).get('seconds'), steady_tok_s=(steady or {}).get('tok_s'),
               steady_turns_per_min=(steady or {}).get('turns_per_min'), tail_s=(tail or {}).get('seconds'),
               tail_tok_s=(tail or {}).get('tok_s'), tail_turns_per_min=(tail or {}).get('turns_per_min'))
    thinking = next((c.get('detail') for c in summary.get('checks') or () if isinstance(c, dict)
                     and c.get('check') == 'thinking' and isinstance(c.get('detail'), dict)), None) or {}
    if thinking.get('thinking'):  # keep-rebuild-1116: the rows' rollbacks and rebuilds by their history
        histories = thinking.get('histories') or {}
        row.update(rollbacks=histories.get('rollback', 0), thinking_rebuilds=sum(
            v for k, v in histories.items() if k.startswith('rebuild')))
    cross = []
    truncated = sum(1 for r in records if r.get('status') == 'truncated')
    if row['server_generated_tokens'] is not None and server['kind'] == 'runtime' and not truncated:
        cross.append(dict(check='numerator-client-equals-server', ok=completion == row['server_generated_tokens'],
                          client=completion, server=row['server_generated_tokens']))
    elif row['server_generated_tokens'] is not None and server['kind'] == 'runtime':
        # killed runners left no (or a partial) trajectory: the server counted replies no client recorded
        cross.append(dict(check='numerator-client-at-most-server-truncated',
                          ok=completion <= row['server_generated_tokens'], client=completion,
                          server=row['server_generated_tokens'], truncated_instances=truncated))
    if row['vllm_generated_tokens'] is not None:
        cross.append(dict(check='numerator-vllm-at-least-client', ok=row['vllm_generated_tokens'] >= completion,
                          client=completion, vllm=row['vllm_generated_tokens']))
    row['numerator_cross_checks'] = cross
    # kernel alignment (swe_checks.check_kernels in the job's and the verifier's server checks, ours and stock alike)
    kernel = next((c for c in summary.get('checks') or () if isinstance(c, dict) and c.get('check') == 'kernel-alignment'),
                  None)
    found = kernel.get('detail') if kernel and isinstance(kernel.get('detail'), dict) else {}
    row['kernel_alignment'] = kernel.get('status') if kernel else 'ABSENT'
    row['kernels'] = '%s/FA%s/GDN-%s/sampler-%s' % (
        ','.join(found.get('attention_backends') or ['?']), ','.join(found.get('flash_attention_versions') or ['?']),
        ','.join(g[1] for g in found.get('gdn_prefill') or ()) or '?', found.get('sampler') or '?') if found else None
    return row


def keep_servers(controller, pins, plan):
    """The plan's thinking-on runtime servers (release gate 1, the keep render: every served prompt)."""
    on = getattr(controller, 'thinking_on', None)
    return [s['label'] for s in controller.plan_servers(pins, plan) if s['kind'] == 'runtime' and on and on(pins, s)]


KEEP_TOOL = 'tools/keep_render_check.py'
KEEP_NOT_AVAILABLE = 'not available (pre-fix runtime)'


def keep_render_sessions(tool, tok_encode, render_template, template, decode, sessions):
    """Release gate 1 (2026-11-16): the sealed runtime's own tools/keep_render_check.py (keep-rebuild-1116) over every
    call of every session: its request messages (the trajectory's, the reasoning the agent sent) against the served
    prompt ids (the runtime rows); the tool's verdict per call (exact_equal AND text_equal). `tool`: the tool's path
    (absent: NOT_AVAILABLE, a runtime before the fix)."""
    if not Path(tool).is_file():
        return dict(status='NOT_AVAILABLE', reason='the sealed runtime has no %s' % KEEP_TOOL)
    import importlib.util
    spec = importlib.util.spec_from_file_location('sealed_keep_render_check', str(tool))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    renderer = module.KeepRenderer(tok_encode, render_template, template, decode)
    results = {}
    for key, calls in sessions.items():
        compared = module.compare(renderer, [c['messages'] for c in calls], [c['prompt'] for c in calls])
        summary = module.summary(compared)
        summary['failures'] = [dict(r) for r in compared if not module.verdict(r)][:5]
        results[key] = summary
    return dict(status='DONE', tool_sha256=hashlib.sha256(Path(tool).read_bytes()).hexdigest(),
                verdict_rule='exact_equal and text_equal', sessions=results)


def keep_render_apply(value, keep_labels, specs, result):
    """The keep-render result into the join: per instance `keep-render`, per server `keep-render-<label>` (fails the
    join), or the server's note (not available / not run) without a check."""
    notes = value.setdefault('keep_render', {})
    for label in keep_labels:
        keys = [k for k in specs if k.startswith(label + '/')]
        if result is None:
            notes[label] = 'not run (no --tokenizer-python)'
            continue
        if result.get('status') == 'NOT_AVAILABLE':
            notes[label] = KEEP_NOT_AVAILABLE
            continue
        sessions = result.get('sessions') or {}
        totals = dict(sessions=len(keys), calls=0, failed=0, exact_equal=0, text_equal=0)
        for record in value['instances']:
            key = '%s/%s' % (record['label'], record['instance_id'])
            if record['label'] != label or key not in specs:
                continue
            item = sessions.get(key)
            ok = result.get('status') == 'DONE' and bool(item) and item['calls'] == len(specs[key]) and \
                item['failed'] == 0
            for name in ('calls', 'failed', 'exact_equal', 'text_equal'):
                totals[name] += (item or {}).get(name) or 0
            record['checks'].append(dict(check='keep-render', status='PASS' if ok else 'FAIL',
                                         detail=json.dumps(item or result.get('error'))[:2000]))
            record['join'] = 'PASS' if all(c['status'] in OK_STATUSES for c in record['checks']) else 'FAIL'
        ok = result.get('status') == 'DONE' and totals['calls'] > 0 and totals['failed'] == 0
        notes[label] = dict(totals, status=result.get('status'), tool_sha256=result.get('tool_sha256'),
                            verdict_rule=result.get('verdict_rule'))
        value['checks'].append(dict(check='keep-render-' + label, status='PASS' if ok else 'FAIL',
                                    detail=json.dumps(dict(totals, status=result.get('status'),
                                                           error=result.get('error')))[:2000]))


def join(check_dir, runs_dir, out, tokenizer_python=None, keep_render=None):
    """`keep_render` (tests only): a callable {session key: calls} -> the keep-render result, in place of the
    tokenizer worker's."""
    check_dir, runs_dir, out = Path(check_dir), Path(runs_dir), Path(out)
    out.mkdir(parents=True)
    report = load(check_dir / 'local-verification.json')
    evidence = Path(report['evidence_dir']) / 'files'
    have_bulk = report.get('bulk_present') is not False and (evidence / 'job-output').is_dir() and any(
        (evidence / 'job-output').glob('*/runtime.jsonl'))
    sys.path.insert(0, str(CANDIDATE))
    import controller
    pins = load(CANDIDATE / 'pins.json')
    options = load(evidence / 'options.json')
    run_id = report.get('run_id')
    value = dict(deltaai_verdict=report.get('verdict'), run_id=run_id, plan=options.get('plan'), runs_dir=str(runs_dir),
                 tasks_provisional=pins['tasks']['provisional'], checks=[], instances=[], servers=[],
                 provisional=sorted(controller.cfg(pins).get('provisional') or {}))
    value['checks'].append(dict(check='run-id', status='PASS' if runs_dir.name == run_id else 'FAIL',
                                detail='%s vs %s' % (runs_dir.name, run_id)))
    verdict = str(report.get('verdict'))
    value['checks'].append(dict(check='deltaai-evidence-verified', status='PASS' if verdict.endswith('_VERIFIED')
                                or verdict == 'ARCHIVE_VERIFIED_SLURM_EXIT_UNCONFIRMED'
                                else 'PENDING' if verdict == 'CORE_VERIFIED_BULK_PENDING' else 'FAIL', detail=verdict))
    pool_tasks = load(evidence / controller.pool_file(pins, options['plan']))  # 2026-11-12: the plan's list (32 or 64)
    value['task_count'] = len(pool_tasks)
    specs = {}
    for server in controller.plan_servers(pins, options['plan']):
        label = server['label']
        summary_path = evidence / ('summary-' + label + '.json')
        summary = load(summary_path) if summary_path.exists() else dict(status='NO_SUMMARY')
        pool = runs_dir / label
        tasks = pool_tasks[:server['tasks']]
        if not pool.is_dir():
            value['servers'].append(dict(label=label, policy=server['policy'], status=summary.get('status'),
                                         evaluation='absent', note='no pool on tbench (not run)'))
            continue
        events = [e for e in load_jsonl(evidence / 'job-output' / label / 'sessions.jsonl') if e.get('event') == 'close']
        close_rows = {e['session_index']: e for e in events}
        records = []
        for session, task in enumerate(tasks):
            record, spec = join_instance(evidence, pool, server, session, task['instance_id'], summary, task, close_rows,
                                         have_bulk)
            if spec is not None:
                specs['%s/%s' % (label, task['instance_id'])] = spec
            records.append(record)
        checks, eval_state = evaluation(pool, records, [t['instance_id'] for t in tasks])
        value['checks'] += [dict(c, label=label) for c in checks]
        value['instances'] += records
        running = controller.server_cfg(pins, server)['max_running_sessions'] if server['kind'] == 'runtime' else \
            controller.cfg(pins)['agent']['max_running']  # (controller.pool_spec's K)
        value['servers'].append(server_row(server, records, summary, evidence, pool, eval_state, running))
    keep_labels = keep_servers(controller, pins, options['plan']) if have_bulk else []
    keep_keys = sorted(k for k in specs if k.split('/', 1)[0] in keep_labels)
    keep_result = None
    if keep_render is not None and keep_labels:
        keep_result = keep_render({k: specs[k] for k in keep_keys})
    if tokenizer_python and specs:
        spec_path, result_path = out / 'render-spec.json', out / 'render-result.json'
        with spec_path.open('x', encoding='utf-8', newline='\n') as stream:
            # 2026-11-06: the plan switch thinking (one value for the plan's runtime servers)
            thinking = any(getattr(controller, 'THINKING_KEY', None) and controller.thinking_on(pins, s)
                           for s in controller.plan_servers(pins, options['plan']))
            # 2026-11-16 (release gate 1): the thinking-on servers' sessions for the sealed keep-render tool
            json.dump(dict(commit=pins['commit'], sessions=specs, page_tokens=16,
                           **(dict(thinking=True) if thinking else {}),
                           **(dict(keep_render=keep_keys) if keep_keys and keep_render is None else {})), stream)
        completed = subprocess.run([str(tokenizer_python), '-B', str(Path(__file__).resolve()), '--render-worker',
                                    str(spec_path), str(result_path)], capture_output=True, text=True, timeout=4 * 3600)
        rendered = load(result_path) if result_path.exists() else dict(status='ERROR', error=completed.stderr[-4000:])
        spec_path.unlink()
        value['exact_render'] = {k: v for k, v in rendered.items() if k not in ('sessions', 'keep_render')}
        if keep_keys and keep_render is None:
            keep_result = rendered.get('keep_render') or dict(status='ERROR', error=rendered.get('error') or
                                                              'the render worker returned no keep-render result')
        for record in value['instances']:
            key = '%s/%s' % (record['label'], record['instance_id'])
            if key not in specs:
                record.setdefault('exact', 'not applicable')
                continue
            item = (rendered.get('sessions') or {}).get(key)
            record['exact'] = 'PASS' if item and item['exact'] else 'FAIL'
            record['checks'].append(dict(check='exact-token-render', status=record['exact'],
                                         detail=json.dumps(item or rendered.get('error'))[:2000]))
            record['join'] = 'PASS' if all(c['status'] == 'PASS' for c in record['checks']) else 'FAIL'
    else:
        for record in value['instances']:
            record.setdefault('exact', 'not run')
    if keep_labels:
        keep_render_apply(value, keep_labels, {k: specs[k] for k in keep_keys}, keep_result)
    stock_history_join(value, controller, pins, options, runs_dir, check_dir, out, tokenizer_python)
    for row in value['servers']:
        if 'kernel_alignment' in row:
            value['checks'].append(dict(check='kernel-alignment-' + row['label'],
                                        status='PASS' if row['kernel_alignment'] == 'PASS' else 'FAIL',
                                        detail='%s %s' % (row['kernel_alignment'], row.get('kernels'))))
        if 'edge_coverage' in row:  # 2026-11-16: a scripted server: no failures.jsonl row, the coverage recorded
            value['checks'].append(dict(check='edge-no-failures-' + row['label'],
                                        status='PASS' if row['edge_failures'] == 0 else 'FAIL',
                                        detail=json.dumps(dict(failures=row['edge_failures'],
                                                               coverage=row['edge_coverage'],
                                                               not_exercised=row['edge_not_exercised']))))
        for cross in row.get('numerator_cross_checks') or ():
            value['checks'].append(dict(check=cross['check'] + '-' + row['label'], status='PASS' if cross['ok'] else 'FAIL',
                                        detail=json.dumps(cross)))
    failed = [c for c in value['checks'] if c['status'] not in OK_STATUSES] + \
        [dict(c, label=r['label'], instance_id=r['instance_id']) for r in value['instances'] for c in r['checks']
         if c['status'] not in OK_STATUSES]
    value['failed'] = failed
    value['bulk_present'] = have_bulk
    pending = any(c['status'] == 'PENDING' for r in value['instances'] for c in r['checks']) or \
        any(c['status'] == 'PENDING' for c in value['checks'])
    value['verdict'] = 'JOIN_FAIL' if failed else 'JOIN_PASS_BULK_PENDING' if pending else 'JOIN_PASS'
    with (out / 'join.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=1, default=str)
        stream.write('\n')
    for name, rows, columns in (('instances.csv', value['instances'], COLUMNS), ('servers.csv', value['servers'],
                                                                                 SERVER_COLUMNS)):
        with (out / name).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow([title for _, title in columns])
            writer.writerows([[row.get(key) for key, _ in columns] for row in rows])
    text = ('Join verdict: %s (DeltaAI %s; run %s; plan %s). Task list PROVISIONAL: %s. Still PROVISIONAL: %s.\n\n'
            % (value['verdict'], value['deltaai_verdict'], run_id, value['plan'], value['tasks_provisional'],
               ', '.join(value['provisional']) or 'nothing')
            + markdown(value['servers'], SERVER_COLUMNS) + '\n\n' + markdown(value['instances'], COLUMNS) + '\n\n'
            'Throughput (PROVISIONAL definition): client completion tokens of every call / span; runner span = first '
            'runner start to last runner end (timeline.jsonl, tbench clock; it includes stretches without a model '
            'call: runner start-up, tool calls, close), model span = first model request to last response. An '
            'instance cut at the stop without a trajectory keeps its runner time but has no client tokens: "tok/s '
            '(runner span; server tokens for instances without a trajectory)" adds its session\'s server-generated '
            'tokens (empty where they are unknown: the stock control); "tok/s (runner span)" is client tokens only, '
            'as before. Decode time lost: vLLM\'s ITL sum minus its observations x the clean step at the interval\'s '
            'batch size, as a share of the ITL sum (measure.decode_lost). Accuracy = resolved / planned instances '
            '(empty patch = unresolved). One run: not benchmark evidence by itself (binomial s.e. up to ~%.1f points '
            'at %d tasks).\n\n%s\n'
            % (50.0 / max(1, len(pool_tasks)) ** 0.5, len(pool_tasks), STEADY_NOTE)
            + ''.join('\nKeep render (release gate 1) %s: %s\n' % (label, json.dumps(note, sort_keys=True))
                      for label, note in sorted((value.get('keep_render') or {}).items())))
    (out / 'join.md').write_bytes(text.encode('utf-8'))
    return value


def stock_history_join(value, controller, pins, options, runs_dir, check_dir, out, tokenizer_python):
    """stock-history-kept (2026-11-16, the stock keep-reasoning plans; a candidate controller without stock_kept or a
    plan without the kept history: nothing): with --tokenizer-python, every call of every instance of a keep server
    re-rendered as stock vLLM 0.24.0 renders it (stock_history.py: the request the keep model class sends, vLLM's parse,
    the pinned template with preserve_thinking, the pinned tokenizer of <check dir>/tokenizer): usage.prompt_tokens
    must equal the kept render on every call (instance check), and the server must have calls where the kept and the
    dropped render differ (the check is not vacuous). Without the tokenizer: 'not run'."""
    kept_of = getattr(controller, 'stock_kept', None)
    servers = [s for s in controller.plan_servers(pins, options['plan']) if kept_of and kept_of(pins, s)]
    if not servers:
        return
    import stock_history
    records = {(r['label'], r['instance_id']): r for r in value['instances']}
    spec = dict(tokenizer=str(Path(check_dir) / 'tokenizer'), instances={})
    for server in servers:
        for (label, instance), record in records.items():
            path = runs_dir / label / 'runs' / instance / instance / (instance + '.traj.json')
            if label != server['label'] or not path.is_file():
                continue
            trajectory = load(path)
            calls = [dict(index=c['index'], kind=c['kind'], prompt_tokens=c['usage'].get('prompt_tokens'),
                          completion_tokens=c['usage'].get('completion_tokens'))
                     for c in stock_history.agent_calls(trajectory)]
            spec['instances']['%s/%s' % (label, instance)] = dict(messages=trajectory['messages'], calls=calls)
            # stock-history-append (no tokenizer): with the reasoning kept every prompt after a reply holds the previous
            # prompt and the whole reply, so prompt[k+1] - prompt[k] - completion[k] >= 0 after every reply (stock32t
            # 3583822's conversations rendered kept: min +44 over 1,421 such pairs; as served, reasoning dropped: 266
            # of them negative, postrun/stock_keep_render.py stock)
            growth = stock_history.append_growth(calls)
            record['checks'].append(dict(check='stock-history-append', status='PASS' if growth['negative'] == 0
                                         else 'FAIL', detail=json.dumps(growth)[:2000]))
            record['join'] = 'PASS' if all(c['status'] in OK_STATUSES for c in record['checks']) else 'FAIL'
        pairs = [json.loads(c['detail']) for (label, _), r in records.items() if label == server['label']
                 for c in r['checks'] if c['check'] == 'stock-history-append']
        total = dict(pairs=sum(g['pairs'] for g in pairs), negative=sum(g['negative'] for g in pairs),
                     min=min([g['min'] for g in pairs if g['min'] is not None], default=None))
        value['checks'].append(dict(check='stock-history-append-' + server['label'],
                                    status='PASS' if total['pairs'] > 0 and total['negative'] == 0 else 'FAIL',
                                    detail=json.dumps(total)))
    if not tokenizer_python:
        for server in servers:
            value.setdefault('stock_history', {})[server['label']] = 'not run (no --tokenizer-python)'
        return
    spec_path, result_path = out / 'stock-history-spec.json', out / 'stock-history-result.json'
    with spec_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(spec, stream)
    completed = subprocess.run([str(tokenizer_python), '-B', str(Path(__file__).resolve()), '--stock-history-worker',
                                str(spec_path), str(result_path)], capture_output=True, text=True, timeout=4 * 3600)
    result = load(result_path) if result_path.exists() else dict(status='ERROR', error=completed.stderr[-4000:])
    spec_path.unlink()
    summary = value.setdefault('stock_history', {})
    for server in servers:
        label = server['label']
        items = {k: v for k, v in (result.get('instances') or {}).items() if k.startswith(label + '/')}
        for (row_label, instance), record in records.items():
            if row_label != label:
                continue
            item = items.get('%s/%s' % (label, instance))
            if item is None and ('%s/%s' % (label, instance)) not in spec['instances']:
                continue  # no trajectory: its own checks say so
            ok = bool(item) and item['calls'] > 0 and item['kept_equal'] == item['calls']
            record['checks'].append(dict(check='stock-history-kept', status='PASS' if ok else 'FAIL',
                                         detail=json.dumps(item or result.get('error'))[:2000]))
            record['join'] = 'PASS' if all(c['status'] in OK_STATUSES for c in record['checks']) else 'FAIL'
        totals = {k: sum(v[k] for v in items.values()) for k in ('calls', 'kept_equal', 'dropped_equal',
                                                                  'distinguishable')}
        totals.update(status=result.get('status'), instances=len(items), history='reasoning_kept')
        summary[label] = totals
        ok = result.get('status') == 'DONE' and totals['calls'] > 0 and totals['kept_equal'] == totals['calls'] \
            and totals['distinguishable'] > 0
        value['checks'].append(dict(check='stock-history-kept-' + label, status='PASS' if ok else 'FAIL',
                                    detail=json.dumps(dict(totals, error=result.get('error')))[:2000]))


def stock_history_worker(spec_path, result_path):
    """Runs under the tokenizer Python: stock_history.check_instance over every instance of the spec."""
    spec = load(spec_path)
    try:
        import stock_history
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec['tokenizer'], local_files_only=True)
        template = stock_history.pinned_template(spec['tokenizer'])
        instances = {key: stock_history.check_instance(tok, template, item['messages'], item['calls'])
                     for key, item in spec['instances'].items()}
        value = dict(status='DONE', template_sha256=stock_history.TEMPLATE_SHA256, instances=instances)
    except Exception as exc:
        import traceback
        value = dict(status='ERROR', error='%s: %s' % (type(exc).__name__, exc), traceback=traceback.format_exc())
    with Path(result_path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=1)
        stream.write('\n')
    return value


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['--render-worker']:
        value = render_worker(argv[1], argv[2])
        print('RENDER_WORKER=' + value['status'])
        return 0 if value['status'] == 'DONE' else 1
    if argv[:1] == ['--stock-history-worker']:  # 2026-11-16
        value = stock_history_worker(argv[1], argv[2])
        print('STOCK_HISTORY_WORKER=' + value['status'])
        return 0 if value['status'] == 'DONE' else 1
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('check_dir')
    parser.add_argument('runs_dir')
    parser.add_argument('--tokenizer-python')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    out = args.out or str(Path(args.check_dir).resolve().parent / 'join')
    value = join(args.check_dir, args.runs_dir, out, args.tokenizer_python)
    for row in value['servers']:
        print('SERVER %s %s resolved=%s/%s throughput_runner=%s tok/s span_runner_s=%s completion=%s server=%s vllm=%s '
              'joined=%s evaluation=%s' % (row['label'], row['policy'], row.get('resolved'), row.get('instances'),
                                          row.get('throughput_runner'), row.get('span_runner_s'),
                                          row.get('completion_tokens'), row.get('server_generated_tokens'),
                                          row.get('vllm_generated_tokens'), row.get('joined'), row.get('evaluation')))
    for failure in value['failed'][:20]:
        print('FAILED %s %s %s' % (failure.get('label', ''), failure.get('instance_id', ''), failure['check']))
    print(value['verdict'])
    return 0 if value['verdict'] in ('JOIN_PASS', 'JOIN_PASS_BULK_PENDING') else 1


if __name__ == '__main__':
    sys.exit(main())
