"""The 150-instance runs side by side (gate 2 analysis; read only on its inputs; standard library only): per job and
per server, and stock control vs full vs ledger_query, the resolved count, the per-instance job completion times (JCT)
and the pool's makespan, throughput (generated tokens / s, requests / s), tokens per turn, turns per instance, format
errors and rollbacks, CAPACITY_WAITs, idle release (evictions / readmissions), ledger_query's active-selection stats,
and a per-instance table with the jobs' columns side by side. EVERY NUMBER IS NOT AUDITED: a number enters group use
only after the job's verified pass and the three release gates of docs/release-gates.md section 3 (keep render on
every call, full against stock token for token, pool invariants after every close); this tool audits nothing.

Usage:
  python3 -B run150_tables.py --out <new dir> <job> [<job> ...] [--core RUN_ID=PATH ...] [--no-evidence]
  python3 -B run150_tables.py --out <new dir> --run150 [--verify-root /home/ubuntu/swe32-verify]

<job>: a verified job directory on tbench (/home/ubuntu/swe32-verify/out/<run_id>/<jobdir>, holding join-*/join.json,
check/checks-<label>.json, tables-*/table1.csv and, for ledger_query, ledger-*/ledger.json) or its run directory
(out/<run_id>, whose RESULT.json names the job directory). The core evidence archive
(<verify root>/evidence/<run_id>/swe32-core.tar.xz, found next to out/, or --core RUN_ID=PATH) is streamed once and only
job-output/<label>/{capacity.jsonl, engine-cleanup.json} (and measurements.jsonl / summary-<label>.json when the check
file lacks the per-request rows) are read into memory; nothing is extracted to disk and the bulk archive is never read.
--run150: exactly the three 150-instance runs (swe32-3616705-r1 stockkeep150k24 = stock vLLM 0.24.0 control,
swe32-3616690-r1 full150k24 = policy full, swe32-3616729-r2 ledger150k24 = ledger_query), refused while any of them
has no verified job directory with a join.

Sources (the join is the authority; the evidence only where the join has no per-instance value):
- join.json (join_swe32.py): resolved (evaluation), per instance wall_s / start / end (runner timeline, tbench clock),
  calls, completion_tokens and prompt_tokens_total (client usage), format_errors, call_tokens; per server
  span_runner_s, throughput_runner, turns_per_min, steady_*, rollbacks / thinking_rebuilds, capacity_waits,
  idle_evictions / idle_readmissions / idle_readmitted_pages, selective_turns.
- check/checks-<label>.json (the verifier's own rerun of check_server): `turns` (one per completed request: session,
  instance_id, capacity_waits = CAPACITY_WAIT refusals of the request, queue_wait_s = FIFO + CAPACITY_WAIT retries,
  prompt_tokens, generated_tokens, selection) and one check `s<k>-turn-<t>-rollback` per rollback row (keep-rebuild-1116);
  release-idle-pages (totals, dropped_by_rebuild, pending_at_end), session-pool (release gate 3), no-preemption (stock).
- core archive: capacity.jsonl (one row per eviction / readmission, the session_index evicted or readmitted),
  engine-cleanup.json idle_release (totals; cross-checked), and the first and last vLLM batch samples
  (job-output/<label>/metrics/sample-<n>.prom: vllm:num_preemptions_total at the end, the stock control's preemption
  count; vllm:request_success_total over the sampler window).
- ledger-*/ledger.json (ledger_verify.py): parity, the hook's counts, selection / admission ms, re-read endpoints.
- tables-*/table1.csv (swe32_tables.py): cross-checked against what this tool computes.
Definitions: JCT = an instance's runner wall (runner start to end; it excludes the wait for a free runner);
"finish from pool start" = its end - the pool's first runner start; makespan = first runner start to last runner end
(span_runner_s); generated tok/s = client completion tokens / makespan (swe32_tables 'tok/s (runner span)'); requests/s
= client model calls / makespan (a CAPACITY_WAIT retry is not a request); percentiles nearest rank (measure.percentile).
Outputs (a new directory): summary.md, summary.json, per-job.csv (one row per job), per-instance.csv (one row per
instance, the jobs' columns side by side). A value that cannot be computed is "n/a (<reason>)", never 0.
Exit 0; 2 when refused (missing inputs, --run150 runs absent, output directory exists).
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import swe32_tables  # noqa: E402  (load / markdown: the existing per-server tables' conventions)

AUDIT = 'NOT AUDITED'
BANNER = ('**NOT AUDITED.** Every number below is unaudited analysis output. A number enters group use (status.md, '
          'a group report, a paper) only after the job\'s offline-verified pass AND the three release gates of '
          'docs/release-gates.md section 3: (1) keep render on every call, (2) full against stock token for token on '
          'a short stretch, (3) pool invariants after every close. This tool audits none of them; the gate states '
          'shown are what the job\'s verification recorded, for information only.')
VERIFY_ROOT = '/home/ubuntu/swe32-verify'
# (tag, policy, run_id, plan): the group's three 150-instance runs at 24 sessions (RUN150_READY.txt), stock first
RUN150 = (('stock', 'stock', 'swe32-3616705-r1', 'stockkeep150k24'),
          ('full', 'full', 'swe32-3616690-r1', 'full150k24'),
          ('ledger_query', 'ledger_query', 'swe32-3616729-r2', 'ledger150k24'))
POLICY_ORDER = ('stock', 'full', 'ledger_query')
ROLLBACK_CHECK = re.compile(r'^s(\d+)-turn-(\d+)-rollback$')
PREEMPTED = re.compile(r",\s*(\d+(?:\.\d+)?)\)")
MAX_MEMBER = 400 * 10 ** 6  # bytes read into memory per archive member


class Refused(Exception):
    pass


def na(reason):
    return 'n/a (%s)' % reason


def is_na(value):
    return isinstance(value, str) and value.startswith('n/a')


def num(value):
    return type(value) in (int, float) and not (isinstance(value, float) and math.isnan(value))


def percentile(values, q):
    """Nearest-rank percentile (measure.percentile): the value at rank ceil(q/100 x n)."""
    values = sorted(v for v in values if num(v))
    return values[max(0, math.ceil(q / 100 * len(values)) - 1)] if values else None


def dist(values, reason, digits=1):
    """n / p50 / p90 / max / mean / sum of the numeric values, or n/a (reason) for each when there are none."""
    values = [v for v in values if num(v)]
    if not values:
        return {k: na(reason) for k in ('n', 'p50', 'p90', 'max', 'mean', 'sum')}
    r = lambda v: round(v, digits) if isinstance(v, float) else v  # noqa: E731
    return dict(n=len(values), p50=r(percentile(values, 50)), p90=r(percentile(values, 90)), max=r(max(values)),
                mean=round(sum(values) / len(values), digits), sum=r(sum(values)))


def ratio(a, b, digits=3, reason='denominator missing'):
    if not num(a) or not num(b):
        return na(reason)
    return round(a / b, digits) if b else na('zero denominator')


# ---------------------------------------------------------------------------------------------------------- inputs
def load_json(path):
    return json.loads(Path(path).read_bytes())


def pick_dir(job_dir, prefix, named, needed):
    """The job's join- / tables- / ledger- directory: the one RESULT.json names, else the newest holding `needed`."""
    if named and (job_dir / named / needed).is_file():
        return job_dir / named
    found = sorted(p for p in job_dir.glob(prefix + '-*') if (p / needed).is_file())
    return found[-1] if found else None


def locate(path):
    """(job_dir, run_dir, result) for a job directory or a run directory (out/<run_id>)."""
    path = Path(path)
    if not path.is_dir():
        raise Refused('%s: not a directory' % path)
    result = None
    if (path / 'RESULT.json').is_file():
        result = load_json(path / 'RESULT.json')
        name = Path(str(result.get('job_dir') or '')).name
        job_dir = path / name if name and (path / name).is_dir() else None
        if job_dir is None:
            dirs = sorted(p for p in path.iterdir() if p.is_dir() and any(p.glob('join-*/join.json')))
            if not dirs:
                raise Refused('%s: RESULT.json names %r and no job directory with a join is there' % (
                    path, result.get('job_dir')))
            job_dir = dirs[-1]
        return job_dir, path, result
    if not any(path.glob('join-*/join.json')):
        raise Refused('%s: no join-*/join.json (not a verified job directory)' % path)
    run_dir = path.parent
    if (run_dir / 'RESULT.json').is_file():
        candidate = load_json(run_dir / 'RESULT.json')
        if Path(str(candidate.get('job_dir') or '')).name == path.name:
            result = candidate
    return path, run_dir, result


SAMPLE = re.compile(r'^job-output/([^/]+)/metrics/sample-(\d+)\.prom$')


def read_core(archive, wanted, sample_labels=()):
    """({member name: bytes} of the wanted members, {label: {'first': (index, text), 'last': (index, text)}} of the
    servers' vLLM /metrics batch samples) of a core tar.xz, streamed once (nothing extracted to disk; a sample is a
    few KB and only the first and the last are kept)."""
    found, samples = {}, {}
    with tarfile.open(str(archive), mode='r|xz') as tar:
        for member in tar:
            if not member.isfile():
                continue
            if member.name in wanted:
                if member.size > MAX_MEMBER:
                    found[member.name] = Refused('%s is %d bytes (cap %d)' % (member.name, member.size, MAX_MEMBER))
                    continue
                found[member.name] = tar.extractfile(member).read()
            else:
                hit = SAMPLE.match(member.name)
                if hit and hit.group(1) in sample_labels and member.size < 10 ** 6:
                    index, slot = int(hit.group(2)), samples.setdefault(hit.group(1), {})
                    first = 'first' not in slot or index < slot['first'][0]
                    last = 'last' not in slot or index > slot['last'][0]
                    if first or last:  # a streamed member can be read once
                        text = tar.extractfile(member).read().decode('utf-8', 'replace')
                        if first:
                            slot['first'] = (index, text)
                        if last:
                            slot['last'] = (index, text)
            if len(found) == len(wanted) and not sample_labels:
                break
    return found, samples


def counter(text, name):
    """The sum over the series of one Prometheus counter family in a /metrics text, or None when it is absent."""
    values = []
    for line in (text or '').splitlines():
        if line.startswith(name + '{') or line.startswith(name + ' '):
            try:
                values.append(float(line.rsplit(' ', 1)[1]))
            except ValueError:
                pass
    return sum(values) if values else None


def jsonl(data):
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def table1(tables_dir):
    """swe32_tables.py's table1.csv rows (titles as keys), or []."""
    if tables_dir is None or not (tables_dir / 'table1.csv').is_file():
        return []
    with (tables_dir / 'table1.csv').open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def check_of(checks, name):
    return next((c for c in (checks or {}).get('checks') or () if c.get('check') == name), None)


def turns_of(rows, source):
    """The per-request rows normalised: session, instance_id, capacity_waits, queue_wait_s, prompt, generated,
    active (an active selection: selection reason null)."""
    out = []
    for r in rows:
        selection = r.get('selection') if isinstance(r.get('selection'), dict) else None
        if selection is not None:
            active = selection.get('reason') is None and selection.get('active', True) is not False
        elif 'selection_active' in r:
            active = r.get('selection_active') is True
        else:
            active = None
        out.append(dict(session=r.get('session'), instance_id=r.get('instance_id'),
                        capacity_waits=r.get('capacity_waits'), queue_wait_s=r.get('queue_wait_s'),
                        prompt=r.get('prompt_tokens'),
                        generated=r.get('generated_tokens', r.get('gen_tokens')), active=active,
                        selected=(selection or {}).get('selected'), candidates=(selection or {}).get('candidate_pages'),
                        source=source))
    return out


# ------------------------------------------------------------------------------------------------------- one server
def server_tables(ctx, server, records, checks, core, ledger, t1):
    """(metrics, per-instance {instance_id: {column: value}}, notes, cross-checks) of one server of one job."""
    label, policy = server.get('label'), server.get('policy')
    stock = policy == 'stock'
    notes, cross = [], []
    xc = lambda name, ours, ref, ok=None: cross.append(dict(  # noqa: E731
        check=name, ours=ours, reference=ref, ok=(ours == ref) if ok is None else ok))
    m = dict(run_id=ctx['run_id'], job=ctx['job_id'], plan=ctx['plan'], label=label, policy=policy, audit=AUDIT,
             deltaai_verdict=ctx['deltaai_verdict'], tbench_verdict=ctx['tbench_verdict'],
             join_verdict=ctx['join_verdict'], join_dir=ctx['join_dir'])
    planned = len(records)
    done = [r for r in records if r.get('status') == 'done']
    m.update(instances=planned, instances_done=len(done),
             instances_truncated=sum(1 for r in records if r.get('status') == 'truncated'))
    # --- resolved
    if server.get('evaluation') == 'present' and num(server.get('resolved')):
        m['resolved'] = server['resolved']
        m['accuracy_pct'] = round(100.0 * server['resolved'] / planned, 2) if planned else na('no instances')
        ours = sum(1 for r in records if r.get('resolved') is True)
        xc('resolved = instances resolved', ours, server['resolved'])
        m['evaluated'] = sum(1 for r in records if r.get('eval_status') is not None)
    else:
        m['resolved'] = m['accuracy_pct'] = m['evaluated'] = na('evaluation %s' % server.get('evaluation'))
    # --- JCT and makespan
    walls = [r.get('wall_s') for r in records]
    starts = [r['start'] for r in records if num(r.get('start'))]
    ends = [r['end'] for r in records if num(r.get('end'))]
    jct = dist(walls, 'no runner timeline in the join')
    for k in ('n', 'p50', 'p90', 'max', 'mean', 'sum'):
        m['jct_s_' + k] = jct[k]
    first = min(starts) if starts else None
    finish = [r['end'] - first for r in records if num(r.get('end')) and first is not None]
    fin = dist(finish, 'no runner timeline in the join')
    for k in ('p50', 'p90', 'max'):
        m['finish_from_pool_start_s_' + k] = fin[k]
    span = server.get('span_runner_s')
    m['makespan_s'] = span if num(span) else na('join has no runner span')
    if starts and ends and num(span):
        xc('makespan = max(end) - min(start)', round(max(ends) - min(starts), 3), round(span, 3),
           abs(max(ends) - min(starts) - span) < 0.01)
    # --- throughput
    calls = sum(r.get('calls') or 0 for r in records)
    completion = sum(r.get('completion_tokens') or 0 for r in records)
    prompt_total = sum(r.get('prompt_tokens_total') or 0 for r in records)
    m.update(calls=calls, gen_tokens_client=completion, prompt_tokens_client=prompt_total)
    xc('calls = sum of instance calls', calls, server.get('calls'))
    xc('client completion tokens', completion, server.get('completion_tokens'))
    m['gen_tok_per_s'] = server.get('throughput_runner') if num(server.get('throughput_runner')) else \
        na('join has no runner throughput')
    if num(span) and span:
        xc('gen tok/s = tokens / makespan', round(completion / span, 2), server.get('throughput_runner'))
    m['requests_per_s'] = ratio(calls, span, 4, 'no runner span')
    m['turns_per_min'] = server.get('turns_per_min') if num(server.get('turns_per_min')) else na('join has none')
    steady = server.get('steady') if isinstance(server.get('steady'), dict) else None
    if steady is not None:
        m['steady_s'] = steady.get('seconds')
        m['gen_tok_per_s_steady'] = steady.get('tok_s')
        m['requests_per_s_steady'] = ratio(steady.get('calls'), steady.get('seconds'), 4)
    elif 'steady' in server:
        m['steady_s'] = m['gen_tok_per_s_steady'] = m['requests_per_s_steady'] = \
            na('the pool never held K = %s sessions in flight' % server.get('running'))
    else:
        m['steady_s'] = m['gen_tok_per_s_steady'] = m['requests_per_s_steady'] = \
            na('join predates the steady-state columns (2026-11-16)')
    # --- tokens per turn, turns per instance
    m['prompt_tok_per_turn_mean'] = ratio(prompt_total, calls, 1, 'no calls')
    m['gen_tok_per_turn_mean'] = ratio(completion, calls, 1, 'no calls')
    turns = None
    if checks is not None and isinstance(checks.get('turns'), list):
        turns = turns_of(checks['turns'], 'check/checks-%s.json turns' % label)
    elif core.get('measurements') is not None:
        turns = turns_of(core['measurements'], 'core measurements.jsonl')
    no_rows = 'stock vLLM: no runtime rows' if stock else (
        'no per-request rows (check file without turns, no core measurements.jsonl)')
    if turns is not None:
        notes.append('per-request rows: %s (%d rows)' % (turns[0]['source'] if turns else 'empty', len(turns)))
    pd_ = dist([t['prompt'] for t in turns] if turns else [], no_rows if not turns else 'no prompt tokens', 0)
    call_gen = [tok for r in records for _, tok in (r.get('call_tokens') or ())]
    if call_gen:
        gd = dist(call_gen, '', 0)
        notes.append('generated tokens per call: client usage (join call_tokens)')
    else:
        gd = dist([t['generated'] for t in turns] if turns else [],
                  'join without call_tokens (before 2026-11-16) and ' + no_rows if not turns else 'none', 0)
        if turns:
            notes.append('generated tokens per call: server rows (join has no call_tokens)')
    for k in ('p50', 'p90', 'max'):
        m['prompt_tok_per_call_' + k] = pd_[k]
        m['gen_tok_per_call_' + k] = gd[k]
    td = dist([r.get('calls') for r in records], 'no calls in the join', 1)
    for k in ('p50', 'p90', 'max', 'mean'):
        m['turns_per_instance_' + k] = td[k]
    # --- format errors, rollbacks
    m['format_errors'] = sum(r.get('format_errors') or 0 for r in records)
    m['instances_with_format_errors'] = sum(1 for r in records if (r.get('format_errors') or 0) > 0)
    thinking = (check_of(checks, 'thinking') or {}).get('detail')
    thinking = thinking if isinstance(thinking, dict) else {}
    rollback_per_session = {}
    rollback_source = None
    if stock:
        m['rollbacks'] = na('stock vLLM: no runtime rows (the agent re-sends the whole history)')
        m['thinking_rebuilds'] = na('stock vLLM: no runtime rows')
    else:
        if num(server.get('rollbacks')):
            m['rollbacks'] = server['rollbacks']
        elif 'rollbacks' in thinking:
            m['rollbacks'] = thinking['rollbacks']
        elif thinking.get('histories'):
            m['rollbacks'] = na('runtime before keep-rebuild-1116: no rollback rows; format errors re-prefill as '
                                'rebuilds (%s)' % (thinking['histories'].get('rebuild')))
        else:
            m['rollbacks'] = na('no thinking record (thinking off or no check file)')
        m['thinking_rebuilds'] = server['thinking_rebuilds'] if num(server.get('thinking_rebuilds')) else (
            sum(v for k, v in (thinking.get('histories') or {}).items() if k.startswith('rebuild'))
            if thinking.get('histories') else na('no thinking record'))
        source_checks = checks if checks is not None else core.get('summary')
        if source_checks is not None and not is_na(m['rollbacks']):
            for c in source_checks.get('checks') or ():
                found = ROLLBACK_CHECK.match(str(c.get('check')))
                if found:
                    k = int(found.group(1))
                    rollback_per_session[k] = rollback_per_session.get(k, 0) + 1
            rollback_source = 'check/checks-%s.json' % label if checks is not None else 'core summary-%s.json' % label
            xc('rollbacks = per-request rollback checks', sum(rollback_per_session.values()), m['rollbacks'])
    # --- CAPACITY_WAIT
    # vLLM's own counters in the server's batch samples (core archive): preemptions at the last sample, finished
    # requests over the sampler window
    samples = core.get('samples')
    if samples:
        last = counter(samples['last'][1], 'vllm:num_preemptions_total')
        m['vllm_preemptions'] = int(last) if last is not None else na('the last sample has no preemption counter')
        a, b = (counter(samples[k][1], 'vllm:request_success_total') for k in ('first', 'last'))
        m['vllm_requests_finished'] = int(b - a) if a is not None and b is not None else na(
            'no vllm:request_success_total in the samples')
        if num(m['vllm_requests_finished']):  # a runner cut at the stop may leave one served, unsaved reply
            xc('vLLM requests finished = client calls (+ truncated)', m['vllm_requests_finished'], calls,
               0 <= m['vllm_requests_finished'] - calls <= m['instances_truncated'])
    else:
        m['vllm_preemptions'] = m['vllm_requests_finished'] = na('core archive not read or without batch samples')
    if stock:
        m['capacity_waits'] = m['capacity_wait_requests'] = m['queue_wait_s_sum'] = \
            na('stock vLLM: no CAPACITY_WAIT gate (it preempts instead)')
        pre = check_of(checks, 'no-preemption')
        if pre is None:
            m['stock_preemptions'] = na('no no-preemption check in the check file')
        elif pre.get('status') == 'PASS':
            m['stock_preemptions'] = 0
        else:
            # the check quotes only the first three scrapes with a nonzero counter: a lower bound
            counts = [float(v) for v in PREEMPTED.findall(str(pre.get('detail')))]
            m['stock_preemptions'] = (m['vllm_preemptions'] if num(m['vllm_preemptions']) else
                                      na('core samples not read; the check quotes at least %d' % max(counts))
                                      if counts else na('preemption %s: %s' % (pre.get('status'),
                                                                               str(pre.get('detail'))[:120])))
            if counts and num(m['vllm_preemptions']):
                xc('vLLM preemptions (last sample) >= the check\'s quoted counters', m['vllm_preemptions'],
                   int(max(counts)), m['vllm_preemptions'] >= max(counts))
            notes.append('stock preemptions: %s %s; the final count is the last batch sample\'s counter' % (
                pre.get('status'), pre.get('finding')))
        if pre is not None and pre.get('status') == 'PASS' and num(m['vllm_preemptions']):
            xc('vLLM preemptions (last sample) = 0 (no-preemption PASS)', m['vllm_preemptions'], 0)
    else:
        m['stock_preemptions'] = na('runtime server: CAPACITY_WAIT, not vLLM preemption')
        m['capacity_waits'] = server['capacity_waits'] if num(server.get('capacity_waits')) else \
            na('join has no CAPACITY_WAIT count')
        if turns:
            per_turn = sum(t['capacity_waits'] or 0 for t in turns)
            m['capacity_wait_requests'] = sum(1 for t in turns if (t['capacity_waits'] or 0) > 0)
            m['queue_wait_s_sum'] = round(sum(t['queue_wait_s'] or 0 for t in turns), 1)
            if num(m['capacity_waits']):
                xc('CAPACITY_WAIT = sum over completed requests', per_turn, m['capacity_waits'])
        else:
            m['capacity_wait_requests'] = m['queue_wait_s_sum'] = na(no_rows)
    stall = check_of(checks, 'capacity-stall')
    m['capacity_stalls'] = (stall.get('detail') or {}).get('stalls') if stall and isinstance(
        stall.get('detail'), dict) else na('stock vLLM' if stock else 'no capacity-stall check')
    # --- idle release
    release = check_of(checks, 'release-idle-pages')
    detail = release.get('detail') if release else None
    on = None
    if stock:
        reason = 'stock vLLM: no --release-idle-pages'
    elif isinstance(detail, dict):
        on = detail.get('release_idle_pages') is True
        reason = None if on else 'server without --release-idle-pages'
    elif 'idle_evictions' in server:
        on, reason = True, None
    else:
        reason = 'release-idle-pages check %s and the join has no idle-release totals' % (
            release.get('status') if release else 'absent')
    events = core.get('capacity')
    if on:
        m['idle_release'] = 'on'
        m['evictions'] = server.get('idle_evictions', na('join has no idle-release totals'))
        m['readmissions'] = server.get('idle_readmissions', na('join has no idle-release totals'))
        m['readmitted_pages'] = server.get('idle_readmitted_pages', na('join has no idle-release totals'))
        if isinstance(detail, dict):
            m['evicted_pages'] = detail.get('evicted_pages', na('check detail without evicted_pages'))
            m['dropped_by_rebuild'] = detail.get('dropped_by_rebuild', 0)
            m['pending_at_end'] = detail.get('pending_at_end', na('check detail without pending_at_end'))
        else:
            reason2 = 'release-idle-pages check %s: %s' % (release.get('status') if release else 'absent',
                                                            str(detail)[:100])
            m['evicted_pages'] = (core.get('cleanup') or {}).get('evicted_pages', na(reason2))
            m['dropped_by_rebuild'] = m['pending_at_end'] = na(reason2)
        if events is not None:
            evs = sum(1 for e in events if e.get('event') == 'evict')
            rds = sum(1 for e in events if e.get('event') == 'readmit')
            xc('evictions = capacity.jsonl evict rows', evs, m['evictions'])
            xc('readmissions = capacity.jsonl readmit rows', rds, m['readmissions'])
        if core.get('cleanup') is not None and events is not None:
            xc('engine-cleanup idle_release = capacity.jsonl totals', core['cleanup'], dict(
                evictions=sum(1 for e in events if e.get('event') == 'evict'),
                evicted_pages=sum(e.get('pages') or 0 for e in events if e.get('event') == 'evict'),
                readmissions=sum(1 for e in events if e.get('event') == 'readmit'),
                readmitted_pages=sum(e.get('pages') or 0 for e in events if e.get('event') == 'readmit')))
    else:
        m['idle_release'] = 'off' if on is False else na(reason)
        for k in ('evictions', 'readmissions', 'readmitted_pages', 'evicted_pages', 'dropped_by_rebuild',
                  'pending_at_end'):
            m[k] = na(reason)
    # --- selection (selective servers) and ledger_query's stats
    m['selective_turns'] = server['selective_turns'] if num(server.get('selective_turns')) else na(
        'stock vLLM' if stock else 'join has no selective_turns')
    active_known = turns and any(t['active'] is not None for t in turns)
    m['active_selection_turns'] = sum(1 for t in turns if t['active']) if active_known else na(
        'stock vLLM' if stock else no_rows)
    if active_known and checks is not None and num(checks.get('active_selection_turns')):
        xc('active selection turns = check file total', m['active_selection_turns'], checks['active_selection_turns'])
    if policy == 'ledger_query':
        entry = ((ledger or {}).get('servers') or {}).get(label)
        if entry is None:
            m['ledger'] = na('ledger.json absent' if ledger is None else 'ledger.json has no server %s' % label)
        else:
            end, parity = entry.get('endpoints') or {}, entry.get('parity') or {}
            hook = end.get('hook') or {}
            stats = parity.get('stats') or {}
            m['ledger'] = dict(
                status=ledger.get('status'), parity=parity.get('status'), parity_requests=stats.get('requests'),
                parity_equal=stats.get('equal'), parity_differs=stats.get('differs'), inactive=stats.get('inactive'),
                hook_requests=hook.get('requests'), ledger_pages_mean=hook.get('ledger_pages_mean'),
                room_pages_mean=hook.get('room_pages_mean'), excluded_pages_mean=hook.get('excluded_pages_mean'),
                items=hook.get('items'), unfilled_requests_pct=hook.get('unfilled_requests_pct'),
                requests_with_skipped_pct=hook.get('requests_with_skipped_pct'), code_pct=hook.get('code_pct'),
                plain_code_pct=hook.get('plain_code_pct'), reasoning_pct=hook.get('reasoning_pct'),
                plain_reasoning_pct=hook.get('plain_reasoning_pct'), plain_jaccard_mean=hook.get('plain_jaccard_mean'),
                selection_ms=end.get('selection_ms'), admission_engine_ms=end.get('admission_engine_ms'),
                steps_per_instance=end.get('steps_per_instance'), reread_steps_pct=end.get('reread_steps_pct'),
                rereads_unattended_pct=end.get('rereads_unattended_pct'), exact_dup_pct=end.get('exact_dup_pct'),
                loop_streaks=end.get('loop_streaks'), loop_instances=end.get('loop_instances'))
            if num(stats.get('requests')) and not is_na(m['active_selection_turns']):
                xc('ledger parity requests = active selection turns', stats.get('requests'),
                   m['active_selection_turns'])
            if num(end.get('resolved')) and num(m.get('resolved')):
                xc('ledger resolved = join resolved', end.get('resolved'), m['resolved'])
    else:
        m['ledger'] = na('policy %s has no ledger hook' % policy)
    # --- release gates (information only: the run is NOT AUDITED either way)
    keep = (ctx['keep_render'] or {}).get(label)
    if stock:
        m['gate1_keep_render'] = na('stock vLLM (gate 1 checks the runtime\'s served prompts)')
    elif isinstance(keep, dict):
        m['gate1_keep_render'] = 'ran: %s/%s calls exact (failed %s)' % (keep.get('exact_equal'), keep.get('calls'),
                                                                        keep.get('failed'))
    elif keep:
        m['gate1_keep_render'] = str(keep)
    else:
        m['gate1_keep_render'] = 'not recorded in join.json'
    m['gate2_full_vs_stock'] = 'not run (deferred, release-gates.md section 3)'
    pool = check_of(checks, 'session-pool')
    m['gate3_pool_invariants'] = na('stock vLLM') if stock else (
        'check session-pool %s' % pool.get('status') if pool else 'no session-pool check in the check file')
    # --- the existing tables (swe32_tables.py table1.csv) against ours
    row = next((r for r in t1 if r.get('Server') == label), None)
    if row is not None:
        for title, ours in (('Resolved', m['resolved']), ('Calls', calls), ('Span s', span),
                            ('Gen tok (client)', completion), ('tok/s (runner span)', m['gen_tok_per_s'])):
            ref = row.get(title)
            try:
                ok = abs(float(ref) - float(ours)) < 0.01
            except (TypeError, ValueError):
                ok = str(ref) == str(ours)
            cross.append(dict(check='table1.csv %s' % title, ours=ours, reference=ref, ok=ok))
    else:
        notes.append('no tables-*/table1.csv row for %s: nothing to cross-check against' % label)
    # --- per instance
    by_session = {r.get('session'): r['instance_id'] for r in records}
    turn_agg = {}
    mismatched = 0
    for t in turns or ():
        instance = by_session.get(t['session'])
        if t['instance_id'] is not None and instance is not None and t['instance_id'] != instance:
            mismatched += 1
        a = turn_agg.setdefault(t['instance_id'] or instance, dict(cw=0, cwr=0, qw=0.0, active=0, rows=0))
        a['cw'] += t['capacity_waits'] or 0
        a['cwr'] += 1 if (t['capacity_waits'] or 0) > 0 else 0
        a['qw'] += t['queue_wait_s'] or 0
        a['active'] += 1 if t['active'] else 0
        a['rows'] += 1
    if turns:
        xc('per-request rows name the session\'s instance', mismatched, 0)
    ev_by, rd_by = {}, {}
    for e in events or ():
        target = ev_by if e.get('event') == 'evict' else rd_by if e.get('event') == 'readmit' else None
        if target is not None:
            target[e.get('session_index')] = target.get(e.get('session_index'), 0) + 1
    if events is not None and on:
        xc('capacity.jsonl sessions are the server\'s instances',
           sorted(k for k in set(ev_by) | set(rd_by) if k not in by_session), [])
    per = {}
    for r in records:
        iid, k = r['instance_id'], r.get('session')
        calls_i = r.get('calls') or 0
        a = turn_agg.get(iid)
        row_i = dict(
            resolved={True: 'R', False: '-', None: '?'}.get(r.get('resolved'), '?') if server.get(
                'evaluation') == 'present' else na('evaluation %s' % server.get('evaluation')),
            status=r.get('status'), exit_status=r.get('exit_status'),
            jct_s=r.get('wall_s') if num(r.get('wall_s')) else na('no runner timeline entry'),
            finish_from_pool_start_s=round(r['end'] - first, 3) if num(r.get('end')) and first is not None else na(
                'no runner end'),
            calls=calls_i, format_errors=r.get('format_errors'),
            gen_tokens=r.get('completion_tokens'), prompt_tokens=r.get('prompt_tokens_total'),
            gen_per_turn=round(r['completion_tokens'] / calls_i, 1) if calls_i and num(r.get('completion_tokens'))
            else na('no calls'),
            prompt_per_turn=round(r['prompt_tokens_total'] / calls_i, 1) if calls_i and num(
                r.get('prompt_tokens_total')) else na('no calls'))
        if stock:
            for key in ('rollbacks', 'capacity_waits', 'capacity_wait_requests', 'queue_wait_s', 'evicted',
                        'readmitted', 'active_turns'):
                row_i[key] = na('stock vLLM')
        else:
            row_i['rollbacks'] = rollback_per_session.get(k, 0) if rollback_source else m['rollbacks'] if is_na(
                m['rollbacks']) else na('no per-request rollback checks')
            if a is not None:
                row_i.update(capacity_waits=a['cw'], capacity_wait_requests=a['cwr'], queue_wait_s=round(a['qw'], 2),
                             active_turns=a['active'] if active_known else na(no_rows))
            elif turns and not calls_i:  # no call, no request
                row_i.update(capacity_waits=0, capacity_wait_requests=0, queue_wait_s=0.0,
                             active_turns=0 if active_known else na(no_rows))
            elif turns:
                for key in ('capacity_waits', 'capacity_wait_requests', 'queue_wait_s', 'active_turns'):
                    row_i[key] = na('no completed request row for this instance')
            else:
                for key in ('capacity_waits', 'capacity_wait_requests', 'queue_wait_s', 'active_turns'):
                    row_i[key] = na(no_rows)
            if on and events is not None:
                row_i.update(evicted=ev_by.get(k, 0), readmitted=rd_by.get(k, 0))
            elif on:
                row_i['evicted'] = row_i['readmitted'] = na('core archive not read: capacity.jsonl is only there')
            else:
                row_i['evicted'] = row_i['readmitted'] = na(reason or 'idle release off')
        per[iid] = row_i
    if on and events is not None:
        xc('per-instance evictions sum', sum(ev_by.get(r.get('session'), 0) for r in records), m['evictions'])
    return m, per, notes, cross


# ---------------------------------------------------------------------------------------------------------- one job
def job_tables(path, core_path=None, evidence=True, expect=None):
    job_dir, run_dir, result = locate(path)
    named = result or {}
    join_dir = pick_dir(job_dir, 'join', named.get('join_dir'), 'join.json')
    if join_dir is None:
        raise Refused('%s: no join-*/join.json' % job_dir)
    join = swe32_tables.load(join_dir / 'join.json')
    tables_dir = pick_dir(job_dir, 'tables', named.get('tables_dir'), 'table1.csv')
    ledger_dir = pick_dir(job_dir, 'ledger', named.get('ledger_dir'), 'ledger.json')
    ledger = swe32_tables.load(ledger_dir / 'ledger.json') if ledger_dir else None
    run_id = join.get('run_id') or named.get('run_id')
    job_id = named.get('job_id') or (re.search(r'swe32-(\d+)-', str(run_id)) or [None, None])[1]
    ctx = dict(run_id=run_id, job_id=job_id, plan=join.get('plan'), deltaai_verdict=join.get('deltaai_verdict'),
               tbench_verdict=named.get('verdict', na('no RESULT.json for this job directory')),
               join_verdict=join.get('verdict'), join_dir=str(join_dir), keep_render=join.get('keep_render'))
    servers = [s for s in join.get('servers') or () if s.get('label')]
    if expect is not None:
        policies = [s.get('policy') for s in servers]
        if policies != [expect[1]] or join.get('plan') != expect[3]:
            raise Refused('%s: plan %s servers %s, expected plan %s with one %s server' % (
                run_id, join.get('plan'), policies, expect[3], expect[1]))
    # the core archive: next to out/ (<verify root>/evidence/<run_id>/swe32-core.tar.xz) unless given
    if core_path is None and evidence:
        guess = run_dir.parent.parent / 'evidence' / str(run_id) / 'swe32-core.tar.xz'
        core_path = guess if guess.is_file() else None
    checks_by = {}
    for s in servers:
        f = job_dir / 'check' / ('checks-%s.json' % s['label'])
        checks_by[s['label']] = swe32_tables.load(f) if f.is_file() else None
    cores, core_note = {s['label']: {} for s in servers}, None
    if core_path is not None and evidence:
        wanted = {}
        for s in servers:
            if s.get('policy') == 'stock':
                continue
            label = s['label']
            wanted['job-output/%s/capacity.jsonl' % label] = (label, 'capacity')
            wanted['job-output/%s/engine-cleanup.json' % label] = (label, 'cleanup')
            if checks_by[label] is None or not isinstance(checks_by[label].get('turns'), list):
                wanted['job-output/%s/measurements.jsonl' % label] = (label, 'measurements')
            if checks_by[label] is None:
                wanted['summary-%s.json' % label] = (label, 'summary')
        found, samples = read_core(core_path, set(wanted), {s['label'] for s in servers})
        for label, slot in samples.items():
            cores[label]['samples'] = slot
        for name, (label, kind) in wanted.items():
            data = found.get(name)
            if isinstance(data, Refused):
                core_note = str(data)
                continue
            if data is None:
                continue
            if kind == 'capacity' or kind == 'measurements':
                cores[label][kind] = jsonl(data)
            elif kind == 'cleanup':
                cores[label][kind] = json.loads(data).get('idle_release')
            else:
                cores[label][kind] = json.loads(data)
        core_note = core_note or 'read %s (%d of %d wanted members; vLLM samples of %s)' % (
            core_path, len(found), len(wanted), ', '.join('%s %d-%d' % (k, v['first'][0], v['last'][0])
                                                          for k, v in sorted(samples.items())) or 'no server')
    else:
        core_note = 'core archive not read (%s)' % ('--no-evidence' if not evidence else 'not found next to out/')
    t1 = table1(tables_dir)
    out = []
    for s in servers:
        records = [r for r in join.get('instances') or () if r.get('label') == s['label']]
        metrics, per, notes, cross = server_tables(ctx, s, records, checks_by[s['label']], cores[s['label']],
                                                   ledger, t1)
        notes = ['job dir %s' % job_dir, core_note] + notes
        if checks_by[s['label']] is None:
            notes.append('no check/checks-%s.json' % s['label'])
        out.append(dict(metrics=metrics, per_instance=per, notes=notes, cross_checks=cross,
                        tag=None, order=(POLICY_ORDER.index(s.get('policy')) if s.get('policy') in POLICY_ORDER
                                         else len(POLICY_ORDER))))
    return out


# ------------------------------------------------------------------------------------------------------- the output
SIDE = [('resolved', 'Resolved'), ('instances', 'Instances'), ('accuracy_pct', 'Accuracy %'),
        ('instances_done', 'Instances done'), ('instances_truncated', 'Instances truncated'),
        ('jct_s_p50', 'JCT s p50 (runner wall)'), ('jct_s_p90', 'JCT s p90'), ('jct_s_max', 'JCT s max'),
        ('jct_s_mean', 'JCT s mean'), ('jct_s_sum', 'JCT s sum'),
        ('finish_from_pool_start_s_p50', 'Finish from pool start s p50'),
        ('finish_from_pool_start_s_p90', 'Finish from pool start s p90'), ('makespan_s', 'Makespan s (pool)'),
        ('gen_tok_per_s', 'Generated tok/s (makespan)'), ('requests_per_s', 'Requests/s (makespan)'),
        ('turns_per_min', 'Turns/min'), ('steady_s', 'Steady s (K in flight)'),
        ('gen_tok_per_s_steady', 'Generated tok/s (steady)'), ('requests_per_s_steady', 'Requests/s (steady)'),
        ('calls', 'Calls (turns)'), ('gen_tokens_client', 'Generated tok (client)'),
        ('prompt_tokens_client', 'Prompt tok (client, sum)'),
        ('prompt_tok_per_turn_mean', 'Prompt tok/turn mean'), ('gen_tok_per_turn_mean', 'Generated tok/turn mean'),
        ('prompt_tok_per_call_p50', 'Prompt tok/call p50'), ('prompt_tok_per_call_p90', 'Prompt tok/call p90'),
        ('prompt_tok_per_call_max', 'Prompt tok/call max'), ('gen_tok_per_call_p50', 'Generated tok/call p50'),
        ('gen_tok_per_call_p90', 'Generated tok/call p90'), ('gen_tok_per_call_max', 'Generated tok/call max'),
        ('turns_per_instance_p50', 'Turns/instance p50'), ('turns_per_instance_p90', 'Turns/instance p90'),
        ('turns_per_instance_max', 'Turns/instance max'), ('turns_per_instance_mean', 'Turns/instance mean'),
        ('format_errors', 'Format errors'), ('instances_with_format_errors', 'Instances with format errors'),
        ('rollbacks', 'Rollbacks'), ('thinking_rebuilds', 'Rebuilds (thinking history)'),
        ('capacity_waits', 'CAPACITY_WAIT refusals'), ('capacity_wait_requests', 'Requests that waited'),
        ('queue_wait_s_sum', 'Server queue wait s (sum; FIFO + CAPACITY_WAIT)'), ('capacity_stalls', 'Capacity stalls'),
        ('stock_preemptions', 'vLLM preemptions (stock)'),
        ('vllm_preemptions', 'vLLM preemption counter (last batch sample)'),
        ('vllm_requests_finished', 'vLLM requests finished (sampler window)'), ('idle_release', 'Idle release'),
        ('evictions', 'Evictions'), ('readmissions', 'Readmissions'), ('evicted_pages', 'Evicted Pages'),
        ('readmitted_pages', 'Readmitted Pages'), ('dropped_by_rebuild', 'Evicted again after a rebuild'),
        ('pending_at_end', 'Evicted, not readmitted at the end'), ('selective_turns', 'Selective turns'),
        ('active_selection_turns', 'Active selection turns'),
        ('gate1_keep_render', 'Gate 1 keep render (recorded)'), ('gate2_full_vs_stock', 'Gate 2 full vs stock'),
        ('gate3_pool_invariants', 'Gate 3 pool invariants (recorded)'), ('tbench_verdict', 'Verification verdict'),
        ('join_verdict', 'Join'), ('audit', 'Audit')]
LEDGER = [('status', 'ledger_verify status'), ('parity', 'Parity'), ('parity_requests', 'Parity requests'),
          ('parity_equal', 'Equal'), ('parity_differs', 'Differs'), ('inactive', 'Inactive requests'),
          ('hook_requests', 'Hook requests'), ('ledger_pages_mean', 'Ledger Pages mean'),
          ('room_pages_mean', 'Room Pages mean'), ('excluded_pages_mean', 'Excluded Pages mean'),
          ('items', 'Items by state'), ('unfilled_requests_pct', 'Unfilled requests %'),
          ('requests_with_skipped_pct', 'Requests with skipped items %'), ('code_pct', 'Code % (plain cosine)'),
          ('reasoning_pct', 'Reasoning % (plain cosine)'), ('plain_jaccard_mean', 'Jaccard vs plain cosine'),
          ('selection_ms', 'Selection ms'), ('admission_engine_ms', 'Admission engine ms'),
          ('steps_per_instance', 'Steps/instance'), ('reread_steps_pct', 'Re-read steps %'),
          ('rereads_unattended_pct', 'Re-reads unattended %'), ('exact_dup_pct', 'Exact-duplicate commands %'),
          ('loop_streaks', 'Loop streaks'), ('loop_instances', 'Loop instances')]
PER = [('resolved', 'resolved'), ('jct_s', 'jct_s'), ('finish_from_pool_start_s', 'finish_s'), ('calls', 'turns'),
       ('gen_per_turn', 'gen_tok_per_turn'), ('prompt_per_turn', 'prompt_tok_per_turn'), ('gen_tokens', 'gen_tok'),
       ('prompt_tokens', 'prompt_tok'), ('format_errors', 'format_errors'), ('rollbacks', 'rollbacks'),
       ('capacity_waits', 'capacity_waits'), ('capacity_wait_requests', 'capacity_wait_requests'),
       ('queue_wait_s', 'queue_wait_s'), ('evicted', 'evicted'), ('readmitted', 'readmitted'),
       ('active_turns', 'active_turns'), ('exit_status', 'exit_status'), ('status', 'status')]


def show(value):
    if value is None:
        return na('none recorded')
    if isinstance(value, float):
        return ('%.3f' % value).rstrip('0').rstrip('.') if abs(value) < 100 else '%.1f' % value
    if isinstance(value, dict):
        return '; '.join('%s %s' % (k, show(v)) for k, v in value.items())
    return str(value)


def md_table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + ' --- |' * len(header)]
    lines += ['| ' + ' | '.join(str(c).replace('|', '/') for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


def write(out, parts, mode, refusal=None):
    out.mkdir(parents=True)
    tags = [p['tag'] for p in parts]
    # per-job.csv: one row per job (server)
    keys = [k for k, _ in SIDE if k != 'audit'] + ['run_id', 'job', 'plan', 'label', 'policy', 'deltaai_verdict',
                                                   'join_dir', 'ledger', 'audit']
    with (out / 'per-job.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['job_tag'] + keys)
        for p in parts:
            writer.writerow([p['tag']] + [show(p['metrics'].get(k)) if not isinstance(p['metrics'].get(k), str)
                                          else p['metrics'].get(k) for k in keys])
    # per-instance.csv: one row per instance, the jobs side by side
    instances = []
    for p in parts:
        for iid in p['per_instance']:
            if iid not in instances:
                instances.append(iid)
    instances.sort()
    with (out / 'per-instance.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['instance_id'] + ['%s:%s' % (t, title) for t in tags for _, title in PER] + ['audit'])
        for iid in instances:
            row = [iid]
            for p in parts:
                v = p['per_instance'].get(iid)
                row += [na('instance not in this job') if v is None else show(v.get(k)) for k, _ in PER]
            writer.writerow(row + [AUDIT])
    # summary.json
    value = dict(audit=AUDIT, banner=BANNER, mode=mode, tool='postrun/run150_tables.py', order=tags,
                 jobs=[dict(tag=p['tag'], metrics=p['metrics'], notes=p['notes'], cross_checks=p['cross_checks'],
                            per_instance=p['per_instance']) for p in parts],
                 cross_checks_failed=sum(1 for p in parts for c in p['cross_checks'] if not c['ok']))
    with (out / 'summary.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=1, sort_keys=True, default=str)
        stream.write('\n')
    # summary.md
    text = io.StringIO()
    text.write('# 150-run tables: %s (NOT AUDITED)\n\n%s\n\n' % (' vs '.join(tags), BANNER))
    text.write(md_table(['Job', 'Run', 'Plan', 'Policy', 'Server', 'DeltaAI verdict', 'Verification', 'Join'],
                        [[p['tag'], p['metrics']['run_id'], p['metrics']['plan'], p['metrics']['policy'],
                          p['metrics']['label'], p['metrics']['deltaai_verdict'], p['metrics']['tbench_verdict'],
                          p['metrics']['join_verdict']] for p in parts]) + '\n\n')
    for p in parts:
        verdict = str(p['metrics']['tbench_verdict'])
        if not (verdict.startswith('SWE32_PASS') and verdict.endswith('_VERIFIED')) or \
                p['metrics']['join_verdict'] != 'JOIN_PASS':
            text.write('**WARNING: %s is not a verified pass (verification %s, join %s): its numbers are not '
                       'usable even after the gates.**\n\n' % (p['tag'], verdict, p['metrics']['join_verdict']))
    failed = [(p['tag'], c['check']) for p in parts for c in p['cross_checks'] if not c['ok']]
    if failed:
        text.write('**WARNING: %d cross-check(s) failed: %s.**\n\n' % (
            len(failed), '; '.join('%s %s' % f for f in failed)))
    text.write('## Side by side (NOT AUDITED)\n\n')
    text.write(md_table(['Metric'] + tags, [[title] + [show(p['metrics'].get(k)) for p in parts]
                                            for k, title in SIDE]) + '\n\n')
    ledgers = [p for p in parts if isinstance(p['metrics'].get('ledger'), dict)]
    text.write('## ledger_query active selection (ledger.json; NOT AUDITED)\n\n')
    if ledgers:
        text.write(md_table(['Metric'] + [p['tag'] for p in ledgers],
                            [[title] + [show(p['metrics']['ledger'].get(k)) for p in ledgers]
                             for k, title in LEDGER]) + '\n\n')
    else:
        text.write('%s\n\n' % '; '.join('%s: %s' % (p['tag'], p['metrics'].get('ledger')) for p in parts))
    text.write('## Cross-checks (this tool against the join, the check file, the core evidence and table1.csv)\n\n')
    text.write(md_table(['Job', 'Check', 'Ours', 'Reference', 'OK'],
                        [[p['tag'], c['check'], show(c['ours']), show(c['reference']), 'yes' if c['ok'] else 'NO']
                         for p in parts for c in p['cross_checks']]) + '\n\n')
    text.write('## Notes\n\n' + ''.join('- %s: %s\n' % (p['tag'], n) for p in parts for n in p['notes']))
    text.write('\nDefinitions: JCT = runner wall (runner start to end, tbench clock; the wait for a free runner '
               'excluded); finish from pool start = end - the first runner start; makespan = first runner start to '
               'last runner end; generated tok/s = client completion tokens / makespan (swe32_tables "tok/s (runner '
               'span)"); requests/s = client model calls / makespan (CAPACITY_WAIT retries are not requests); steady '
               '= the stretches with K sessions in flight (join_swe32.STEADY_NOTE); prompt tok/call = served prompt '
               'tokens of the runtime\'s completed requests; percentiles nearest rank. Per-instance values: '
               'per-instance.csv (one row per instance, the jobs side by side).\n\n**%s**\n' % AUDIT)
    (out / 'summary.md').write_bytes(text.getvalue().encode('utf-8'))
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('jobs', nargs='*')
    parser.add_argument('--run150', action='store_true', help='the three 150-instance runs under --verify-root')
    parser.add_argument('--verify-root', default=VERIFY_ROOT)
    parser.add_argument('--core', action='append', default=[], help='RUN_ID=PATH of a core archive')
    parser.add_argument('--no-evidence', action='store_true', help='read no core archive')
    args = parser.parse_args(argv)
    out = Path(args.out)
    try:
        if out.exists():
            raise Refused('%s exists (the tool writes a new directory)' % out)
        cores = {}
        for item in args.core:
            run_id, _, path = item.partition('=')
            if not path or not Path(path).is_file():
                raise Refused('--core %s: expected RUN_ID=PATH of an existing archive' % item)
            cores[run_id] = Path(path)
        parts = []
        if args.run150:
            if args.jobs:
                raise Refused('--run150 takes no job directories')
            root = Path(args.verify_root)
            missing = []
            for expect in RUN150:
                run_dir = root / 'out' / expect[2]
                if not (run_dir / 'RESULT.json').is_file():
                    missing.append('%s (%s): %s' % (expect[2], expect[3], 'no %s' % (run_dir / 'RESULT.json')))
                    continue
                try:
                    locate(run_dir)
                except Refused as exc:
                    missing.append('%s (%s): %s' % (expect[2], expect[3], exc))
            if missing:
                raise Refused('RUN150 not ready: the three 150-instance runs must be verified on tbench first; '
                              'absent: ' + '; '.join(missing))
            for expect in RUN150:
                run_dir = root / 'out' / expect[2]
                for p in job_tables(run_dir, cores.get(expect[2]), not args.no_evidence, expect):
                    p['tag'] = expect[0]
                    parts.append(p)
            mode = 'run150'
        else:
            if not args.jobs:
                raise Refused('give job directories or --run150')
            for path in args.jobs:
                run_guess = Path(path).name if (Path(path) / 'RESULT.json').is_file() else Path(path).parent.name
                parts += job_tables(path, cores.get(run_guess), not args.no_evidence)
            parts.sort(key=lambda p: p['order'])
            multi = {}
            for p in parts:
                multi[p['metrics']['job']] = multi.get(p['metrics']['job'], 0) + 1
            for p in parts:
                m = p['metrics']
                p['tag'] = ('%s/%s (%s)' % (m['policy'], m['label'], m['job']) if multi[m['job']] > 1
                            else '%s (%s)' % (m['policy'], m['job']))
            mode = 'jobs'
        value = write(out, parts, mode)
    except Refused as exc:
        print('RUN150_TABLES_REFUSED %s' % exc)
        return 2
    for job in value['jobs']:
        m = job['metrics']
        print('RUN150_TABLES %s resolved=%s/%s jct_p50=%s makespan=%s tok/s=%s req/s=%s rollbacks=%s '
              'capacity_waits=%s evictions=%s readmissions=%s %s' % (
                  job['tag'], show(m['resolved']), m['instances'], show(m['jct_s_p50']), show(m['makespan_s']),
                  show(m['gen_tok_per_s']), show(m['requests_per_s']), show(m['rollbacks']),
                  show(m['capacity_waits']), show(m['evictions']), show(m['readmissions']), AUDIT))
    print('RUN150_TABLES_DONE out=%s jobs=%d cross_checks_failed=%d %s' % (
        out, len(value['jobs']), value['cross_checks_failed'], AUDIT))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
