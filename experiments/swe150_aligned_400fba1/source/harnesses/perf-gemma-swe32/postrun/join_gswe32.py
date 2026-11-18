"""Join the two halves of one perf-gemma-swe32 job locally: DeltaAI server evidence (verified by verify_evidence.py) <->
tbench run (the pools' trajectories, timelines, close receipts, host samples and the SWE-bench evaluation), one row per
instance x server, plus per-server throughput, accuracy and failures.

Usage:
  python -B join_gswe32.py <check dir> <tbench run dir> [--tokenizer-python PY] [--out DIR]

<check dir>: verify_evidence.py's output (local-verification.json; its `evidence_dir` holds the extracted files).
<tbench run dir>: runs/<run_id> from the fetched package: <label>/{instances.txt, timeline.jsonl, closes/, runs/<id>/
<id>/<id>.traj.json, runs/<id>/preds.json, preds.json, host-samples.jsonl, spec.json, model_override.yaml, eval/}.

Per runtime server and session (= the instance's index in the server's task list), pilot-v14's join (join_pilot.py)
for the session pool: the agent's model calls (assistant messages and format errors) against the session's runtime
rows (calls = rows, response ids in order, usage = receipts: prompt tokens N, completion C' - N + 1), every format error
followed by a rebuild, api_calls, preds.json against the submission; the close receipt the pool saved
(closes/<id>.json) equal to the session's close row in sessions.jsonl; the timeline entry; with --tokenizer-python the
exact-token re-render of every request through the sealed release's render path (a fresh clone of the sealed bundle):
gemma4_renderer.Gemma4SourceRenderer over the pinned Gemma 4 metadata the verifier copied into <check dir>/tokenizer,
as the server's live renderer does: the agent <-> server proof. The stock control has no server evidence: its calls and usage only.

Throughput (config.throughput, PROVISIONAL): main-agent completion tokens
of every call of every instance (client usage) / span, span = first runner start to last runner end on the tbench clock
(timeline.jsonl, close included); also the span from the first model request to the last response (trajectory
timestamps). Cross-checks of the numerator: the server's generated tokens (C' - N + 1 per row) and vLLM's
generation_tokens_total delta over the sampler window (it also counts requests cancelled at a stop). Accuracy: resolved /
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
# The pinned Gemma 4 metadata the server renders with (the sealed commit's model_profile.GEMMA4_MODEL_PINS =
# verify_evidence.TOKENIZER_FILES; google/gemma-4-31B-it @ 842da379)
GEMMA4_FILES = {
    'chat_template.jinja': 'ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4',
    'config.json': 'e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1',
    'generation_config.json': 'd4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de',
    'tokenizer.json': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',
    'tokenizer_config.json': '9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633'}
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
                  ('span_model_s', 'Span s (model calls)'), ('throughput_model', 'tok/s (model span)'),
                  ('calls', 'Calls'), ('turns_per_min', 'Turns/min'), ('context_refusals', 'Context refusals'),
                  ('timeouts', 'Timeouts'), ('capacity_waits', 'CAPACITY_WAIT'), ('joined', 'Joined'),
                  ('kernels', 'Kernels (attention/FA/GDN prefill/sampler)')]
MB = 1e6


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
                          request=[dict(role=m['role'], content=m['content']) for m in messages[:index]
                                   if m.get('role') in ('system', 'user', 'assistant')]))
    return calls


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


OK_STATUSES = ('PASS', 'PENDING')  # PENDING: a server-row check that needs gswe32-bulk.tar.xz (fetched later)


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
    record.update(calls=len(calls), format_errors=sum(1 for c in calls if c['kind'] == 'format_error'),
                  exit_status=info.get('exit_status'), api_calls=(info.get('model_stats') or {}).get('api_calls'),
                  patch_chars=len(info.get('submission') or ''),
                  completion_tokens=sum((c['usage'] or {}).get('completion_tokens') or 0 for c in calls),
                  prompt_tokens_total=sum((c['usage'] or {}).get('prompt_tokens') or 0 for c in calls),
                  max_prompt_tokens=max([(c['usage'] or {}).get('prompt_tokens') or 0 for c in calls] or [0]),
                  first_call_ts=min([c['timestamp'] for c in calls if isinstance(c['timestamp'], (int, float))], default=None),
                  last_call_ts=max([c['timestamp'] for c in calls if isinstance(c['timestamp'], (int, float))], default=None))
    messages = trajectory.get('messages') or []
    content = messages[1].get('content') if len(messages) > 1 and isinstance(messages[1], dict) else None
    add('task-message-is-the-sealed-statement', isinstance(content, str) and task['problem_statement'] in content,
        'messages[1] must carry the sealed problem statement of ' + instance)
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
        record['checks'].append(dict(check='server-rows', status='PENDING', detail='gswe32-bulk.tar.xz not fetched yet: '
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
    """Runs under the tokenizer Python: re-render every call of every session with the release's render path for the
    Gemma 4 profile, as the server's live renderer does (remote_ra_linux/live_renderer.py: a Gemma 4 release renders
    the XML protocol with gemma4_renderer.Gemma4SourceRenderer over the pinned model files: chat_template.jinja, the
    system and task as OpenAI text parts with enable_thinking false, the GEMMA4_XML grammar, stop ids {0, 1, 50, 106}).
    The pinned Gemma 4 metadata is the verifier's copy (`gemma_metadata`: <check dir>/tokenizer), each file against
    GEMMA4_FILES and the sealed commit's model_profile.GEMMA4_MODEL_PINS. (Until 2026-10-24 this rendered with the Qwen
    profile's files and EngineSourceRenderer, perf-swe32's: every Gemma call differed from its first token, jobs
    3429808 / 3429830.)"""
    spec = load(spec_path)
    work = Path(tempfile.mkdtemp(prefix='gswe32-join-'))
    results = {}
    try:
        repo = clone_release(work)
        overlay = repo / 'source-overlay'
        model = work / 'model'; model.mkdir()
        source = Path(spec['gemma_metadata'])
        for name, pin in GEMMA4_FILES.items():
            raw = (source / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != pin:
                raise ValueError('pinned Gemma 4 metadata differs from the pin: ' + name)
            (model / name).write_bytes(raw)
        sys.path.append(str(overlay))
        from transformers import AutoTokenizer
        from gemma4_renderer import Gemma4SourceRenderer
        from remote_ra_live_v2.model_profile import GEMMA4_MODEL_PINS
        if dict(GEMMA4_MODEL_PINS) != GEMMA4_FILES:
            raise ValueError('the sealed release pins other Gemma 4 files than this join')
        tok = AutoTokenizer.from_pretrained(str(model), local_files_only=True)

        def render_template(template, conversation, kwargs):
            return tok.apply_chat_template(conversation, chat_template=template, tokenize=False, return_dict=False, **kwargs)
        make = lambda: Gemma4SourceRenderer(model_dir=model, tokenizer=tok, render_template=render_template,
                                            verify_identity=lambda: True, page_tokens=spec.get('page_tokens', 16))
        for key, calls in spec['sessions'].items():
            transcript = make().transcript()
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


def vllm_generated(out):
    """vLLM's generation_tokens_total over the sampler window (last readable sample - first), or None."""
    sys.path.insert(0, str(CANDIDATE))
    import measure
    try:
        timeline = measure.batch_timeline(out)
    except Exception:
        return None
    values = [p['generated'] for p in timeline if p['ok'] and p['generated'] is not None]
    return round(values[-1] - values[0]) if len(values) >= 2 else None


def markdown(rows, columns):
    lines = ['| ' + ' | '.join(title for _, title in columns) + ' |', '|' + ' --- |' * len(columns)]
    lines += ['| ' + ' | '.join('' if row.get(key) is None else str(row.get(key)) for key, _ in columns) + ' |'
              for row in rows]
    return '\n'.join(lines)


def server_row(server, records, summary, evidence, pool, eval_state):
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
    row = dict(label=label, policy=server['policy'], instances=len(records), done=len(done),
               resolved=sum(1 for r in evaluated if r.get('resolved')) if eval_state == 'present' else None,
               completion_tokens=completion, server_generated_tokens=measured.get('gen_tokens'),
               vllm_generated_tokens=vllm_generated(out), span_runner_s=span_runner, span_model_s=span_model,
               throughput_runner=round(completion / span_runner, 2) if span_runner else None,
               throughput_model=round(completion / span_model, 2) if span_model else None,
               calls=sum(r.get('calls') or 0 for r in records),
               turns_per_min=round(sum(r.get('calls') or 0 for r in records) / span_runner * 60, 2) if span_runner else None,
               context_refusals=sum(1 for f in failures if 'maximum context length' in json.dumps(f).lower()),
               timeouts=sum(1 for r in records if 'timeout' in str(r.get('exit_status') or '').lower()),
               capacity_waits=measured.get('capacity_waits'), joined=sum(1 for r in records if r.get('join') == 'PASS'),
               evaluation=eval_state, prompt_tokens=measured.get('prompt_tokens'),
               prompt_buckets=measured.get('prompt_buckets'), admission_stall_fraction=measured.get('admission_stall_fraction'),
               queue_wait_s=measured.get('queue_wait_s'), selective_turns=measured.get('selective_turns'),
               rebuilds=measured.get('rebuilds'), goal=measured.get('goal'), batch=measured.get('batch'))
    row['accuracy'] = round(100.0 * row['resolved'] / row['instances'], 2) if row['resolved'] is not None else None
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
    # Gemma 4 on TRITON_ATTN: no FlashAttention version and no GDN prefill kernel (an aligned check names none: 'none')
    absent = 'none' if row['kernel_alignment'] == 'PASS' else '?'
    row['kernels'] = '%s/FA%s/GDN-%s/sampler-%s' % (
        ','.join(found.get('attention_backends') or ['?']), ','.join(found.get('flash_attention_versions') or [absent]),
        ','.join(g[1] for g in found.get('gdn_prefill') or ()) or absent, found.get('sampler') or '?') if found else None
    return row


def join(check_dir, runs_dir, out, tokenizer_python=None):
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
    pool_tasks = load(evidence / controller.TASK_POOL)
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
        value['servers'].append(server_row(server, records, summary, evidence, pool, eval_state))
    if tokenizer_python and specs:
        spec_path, result_path = out / 'render-spec.json', out / 'render-result.json'
        with spec_path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(dict(commit=pins['commit'], sessions=specs, page_tokens=16,
                           gemma_metadata=str(check_dir / 'tokenizer')), stream)
        completed = subprocess.run([str(tokenizer_python), '-B', str(Path(__file__).resolve()), '--render-worker',
                                    str(spec_path), str(result_path)], capture_output=True, text=True, timeout=4 * 3600)
        rendered = load(result_path) if result_path.exists() else dict(status='ERROR', error=completed.stderr[-4000:])
        spec_path.unlink()
        value['exact_render'] = {k: v for k, v in rendered.items() if k != 'sessions'}
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
    for row in value['servers']:
        if 'kernel_alignment' in row:
            value['checks'].append(dict(check='kernel-alignment-' + row['label'],
                                        status='PASS' if row['kernel_alignment'] == 'PASS' else 'FAIL',
                                        detail='%s %s' % (row['kernel_alignment'], row.get('kernels'))))
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
            'runner start to last runner end (timeline.jsonl, tbench clock), model span = first model request to last '
            'response. Accuracy = resolved / planned instances (empty patch = unresolved). One run: not benchmark '
            'evidence by itself (binomial s.e. up to ~%.1f points at %d tasks).\n'
            % (50.0 / max(1, len(pool_tasks)) ** 0.5, len(pool_tasks)))
    (out / 'join.md').write_bytes(text.encode('utf-8'))
    return value


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['--render-worker']:
        value = render_worker(argv[1], argv[2])
        print('RENDER_WORKER=' + value['status'])
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
