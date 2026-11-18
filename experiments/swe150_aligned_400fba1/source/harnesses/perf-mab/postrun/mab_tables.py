"""Tables of one verified perf-mab archive (local only): `python -B mab_tables.py <check dir>`.

Reads the verifier's extracted files (<check dir>/files) and writes <check dir>/mab-tables/:
- servers.csv / .md: per server its status, start, agent time, cases and requests, clean answers, synthetic answers
  correct, output tokens, ITL by running count (b = 1 and the peak), preemptions, GPU / host memory peaks, route-B
  totals and failures.jsonl rows;
- turns.csv: per request of every server: case, turn, kind, HTTP status, input / prompt / cached / completion tokens,
  finish reason, client seconds, queue / open waits, capacity waits, selected / candidate Pages, restored H2D bytes,
  readmitted Pages and the sessions released to admit it, the applied Goal decision (goal_query: INITIALIZE / KEEP /
  PATCH / REWRITE; the release smoke's REWRITE count);
- cases.csv: per case: status, chunks read, turns, correct (synthetic), output tokens, wall seconds;
- route-b.csv: per server and runtime session: releases, readmissions, Pages, re-releases, and the releases while
  the session's next request waited for admission (queued_releases: sent by the client, not yet readmitted;
  queued_at_server: of these, after the server certainly held it; mab_checks.queued_releases, never a verdict);
  readmissions.csv: each readmission frame's elapsed ms and H2D bytes (from native-summary.jsonl);
- replay.csv (G1c): per turn: prompt and reply tokens, identical, first difference;
- probes.csv (G1d) and tables.md (everything with the definitions).
The tables never change the verdict; they derive from the verified files only.
"""
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.dont_write_bytecode = True
import mab_checks  # noqa: E402
import mab_metrics  # noqa: E402
import resources  # noqa: E402


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if row.get(k) is None else row.get(k)) for k in fields})
    return path


def md_table(rows, fields):
    out = ['| ' + ' | '.join(fields) + ' |', '|' + '---|' * len(fields)]
    for row in rows:
        out.append('| ' + ' | '.join('' if row.get(k) is None else str(row.get(k)) for k in fields) + ' |')
    return '\n'.join(out)


def server_rows(files, result):
    rows = []
    for r in result.get('servers') or ():
        label = r['label']
        out = files / 'job-output' / label
        client = mab_checks.load_client(out) if out.is_dir() else dict(requests=[], cases=[], probes=[], replays=[])
        metrics = mab_metrics.summarize(out) if out.is_dir() else {}
        itl = metrics.get('itl_by_running') or {}
        peak = max((int(b) for b in itl), default=None)
        try:
            res = resources.summarize(files, label) or {}
        except Exception:
            res = {}
        events = mab_checks.jsonl(out / 'capacity.jsonl')
        cases = client['cases']
        rows.append(dict(
            label=label, kind=r.get('kind'), role=r.get('role'), status=r.get('status'), startup_s=r.get('startup_s'),
            agent_seconds=r.get('agent_seconds'), cases=len(cases),
            answered=sum(c['status'] in mab_checks.CLEAN for c in cases),
            correct=sum(bool(c.get('correct')) for c in cases), requests=len(client['requests']),
            output_tokens=sum(c.get('agent_output_tokens') or 0 for c in cases),
            replays=len(client.get('replays') or ()),
            replays_identical=sum(x.get('identical') is True for x in client.get('replays') or ()),
            itl_b1_ms=(itl.get('1') or {}).get('itl_ms'), itl_peak_b=peak,
            itl_peak_ms=(itl.get(str(peak)) or {}).get('itl_ms') if peak else None,
            running_max=metrics.get('running_max'), preemptions=metrics.get('preemptions_max'),
            gpu_mib_max=res.get('gpu_memory_used_mib_max'), host_rss_max=res.get('host_vmrss_bytes_max'),
            cgroup_peak=res.get('cgroup_memory_peak_bytes_max'),
            releases=sum(e['event'] == 'evict' for e in events), readmissions=sum(e['event'] == 'readmit' for e in events),
            released_pages=sum(e['pages'] for e in events if e['event'] == 'evict'),
            failures_rows=len(mab_checks.jsonl(out / 'failures.jsonl')),
            findings='+'.join(sorted({f['finding'] for f in r.get('findings') or ()})),
            failed='+'.join(r.get('failed') or ())))
    return rows


def turn_rows(files, result):
    rows = []
    for r in result.get('servers') or ():
        out = files / 'job-output' / r['label']
        if not out.is_dir():
            continue
        client = mab_checks.load_client(out)
        server = {}
        for row in mab_checks.runtime_rows(out):
            server.setdefault((row.get('kv_session') or {}).get('session_id'), []).append(row)
        for items in server.values():
            items.sort(key=lambda x: x['ordinal'])
        position = {}
        for q in sorted(client['requests'], key=lambda x: x['seq']):
            response = q.get('response') or {}
            srow = None
            if q.get('mode') == 'kv' and q.get('status') == 200:
                k = position.get(q['sid'], 0)
                position[q['sid']] = k + 1
                srow = (server.get(q['sid']) or [None] * (k + 1))[k] if k < len(server.get(q['sid']) or ()) else None
            body = q.get('body') or {}
            release = (srow or {}).get('idle_release') or {}
            phase1 = ((srow or {}).get('stage_b') or {}).get('phase1') or {}
            rows.append(dict(
                server=r['label'], case=q['case'], sid=(q.get('sid') or '')[:8], turn=q['turn'], kind=q['kind'],
                status=q.get('status'), input_tokens=len(body.get('input_ids') or ()) or body.get('prompt_tokens'),
                prompt_tokens=response.get('prompt_tokens'), cached_tokens=response.get('cached_tokens'),
                completion_tokens=response.get('completion_tokens'), finish_reason=response.get('finish_reason'),
                client_s=round((q['t_recv_ns'] - q['t_send_ns']) / 1e9, 3) if q.get('t_recv_ns') and q.get('t_send_ns')
                else None,
                queue_wait_s=response.get('queue_wait_s'), open_wait_s=response.get('open_wait_s'),
                capacity_waits=response.get('capacity_waits'), selected_pages=response.get('selected_pages'),
                candidate_pages=response.get('candidate_pages'), restored_h2d_bytes=response.get('restored_h2d_bytes'),
                readmitted_pages=len(release.get('readmitted_pages') or ()),
                released_sessions=len(release.get('evicted_sessions') or ()),
                phase1_ms=round(phase1['phase1_ns'] / 1e6, 3) if phase1.get('phase1_ns') else None,
                selection_active=(((srow or {}).get('query') or {}).get('selection') or {}).get('reason') is None
                if (srow or {}).get('query') else None,
                # integrate-1103 (the release smoke's REWRITE count): the turn's applied Goal decision (goal_query)
                goal_decision=((((srow or {}).get('query') or {}).get('goal')) or {}).get('applied_decision')))
    return rows


def case_rows(files, result):
    rows = []
    for r in result.get('servers') or ():
        out = files / 'job-output' / r['label']
        if not out.is_dir():
            continue
        for c in mab_checks.load_client(out)['cases']:
            rows.append(dict(server=r['label'], case=c['case'], status=c['status'], chunks_read=c['chunks_read'],
                             total_chunks=c['total_chunks'], turns=c['turns'], correct=c.get('correct'),
                             output_tokens=c.get('agent_output_tokens'), answer_tokens=c.get('answer_output_tokens'),
                             wall_s=c.get('wall_s')))
    return rows


def route_rows(files, result):
    sessions, frames = [], []
    for r in result.get('servers') or ():
        out = files / 'job-output' / r['label']
        events = sorted(mab_checks.jsonl(out / 'capacity.jsonl'), key=lambda e: e['monotonic_ns'])
        try:
            queued = (mab_checks.queued_releases(out, mab_checks.load_client(out), mab_checks.runtime_rows(out))
                      if events else {})
        except mab_checks.CheckFailed:      # a malformed runner.json clock (check_runner / route-b fail on it)
            queued = {}
        per, last = {}, {}
        for e in events:
            s = per.setdefault(e['session_id'], dict(server=r['label'], session=e['session_id'][:12],
                                                    session_index=e.get('session_index'), releases=0, readmissions=0,
                                                    released_pages=0, readmitted_pages=0, re_releases=0,
                                                    queued_releases=(queued.get(e['session_id']) or {}).get('queued', 0),
                                                    queued_at_server=(queued.get(e['session_id']) or {})
                                                    .get('queued_at_server', 0)))
            if e['event'] == 'evict':
                s['releases'] += 1
                s['released_pages'] += e['pages']
                if last.get(e['session_id']) == 'readmit':
                    s['re_releases'] += 1
            else:
                s['readmissions'] += 1
                s['readmitted_pages'] += e['pages']
            last[e['session_id']] = e['event']
        sessions += list(per.values())
        for n in mab_checks.jsonl(out / 'native-summary.jsonl'):
            for f in n.get('readmit') or ():
                frames.append(dict(server=r['label'], request=n['request_id'], session_index=n.get('session_index'),
                                   turn=n.get('turn'), pages=f.get('pages'), h2d_bytes=f.get('h2d_bytes'),
                                   elapsed_ms=round(f['elapsed_ns'] / 1e6, 3) if f.get('elapsed_ns') else None,
                                   gb_per_s=round(f['h2d_bytes'] / f['elapsed_ns'], 3)
                                   if f.get('elapsed_ns') and f.get('h2d_bytes') else None,
                                   admission_engine_ms=n.get('admission_engine_ms')))
    return sessions, frames


def write_tables(check):
    check = Path(check)
    files = check / 'files'
    result = json.loads((files / 'result.json').read_bytes())
    target = check / 'mab-tables'
    target.mkdir(exist_ok=True)
    written = []
    servers = server_rows(files, result)
    fields = ['label', 'kind', 'role', 'status', 'startup_s', 'agent_seconds', 'cases', 'answered', 'correct',
              'requests', 'output_tokens', 'replays', 'replays_identical', 'itl_b1_ms', 'itl_peak_b', 'itl_peak_ms',
              'running_max', 'preemptions', 'gpu_mib_max', 'host_rss_max', 'cgroup_peak', 'releases', 'readmissions',
              'released_pages', 'failures_rows', 'findings', 'failed']
    written.append(write_csv(target / 'servers.csv', servers, fields))
    turns = turn_rows(files, result)
    written.append(write_csv(target / 'turns.csv', turns, ['server', 'case', 'sid', 'turn', 'kind', 'status',
                                                           'input_tokens', 'prompt_tokens', 'cached_tokens',
                                                           'completion_tokens', 'finish_reason', 'client_s',
                                                           'queue_wait_s', 'open_wait_s', 'capacity_waits',
                                                           'selected_pages', 'candidate_pages', 'restored_h2d_bytes',
                                                           'readmitted_pages', 'released_sessions', 'phase1_ms',
                                                           'selection_active', 'goal_decision']))
    cases = case_rows(files, result)
    written.append(write_csv(target / 'cases.csv', cases, ['server', 'case', 'status', 'chunks_read', 'total_chunks',
                                                           'turns', 'correct', 'output_tokens', 'answer_tokens',
                                                           'wall_s']))
    sessions, frames = route_rows(files, result)
    written.append(write_csv(target / 'route-b.csv', sessions, ['server', 'session', 'session_index', 'releases',
                                                               'readmissions', 'released_pages', 'readmitted_pages',
                                                               're_releases', 'queued_releases', 'queued_at_server']))
    written.append(write_csv(target / 'readmissions.csv', frames, ['server', 'request', 'session_index', 'turn',
                                                                   'pages', 'h2d_bytes', 'elapsed_ms', 'gb_per_s',
                                                                   'admission_engine_ms']))
    replays, probes = [], []
    for r in result.get('servers') or ():
        out = files / 'job-output' / r['label']
        if out.is_dir():
            client = mab_checks.load_client(out)
            replays += [dict(server=r['label'], **{k: x.get(k) for k in ('case', 'turn', 'prompt_tokens',
                                                                          'expected_tokens', 'status', 'identical',
                                                                          'first_difference', 'finish_reason')})
                        for x in client['replays']]
            probes += [dict(server=r['label'], name=p['name'], ok=p['ok'], status=p.get('status'),
                            detail=json.dumps(p.get('body') or p.get('error') or '')[:200]) for p in client['probes']]
    written.append(write_csv(target / 'replay.csv', replays, ['server', 'case', 'turn', 'prompt_tokens',
                                                             'expected_tokens', 'status', 'identical',
                                                             'first_difference', 'finish_reason']))
    written.append(write_csv(target / 'probes.csv', probes, ['server', 'name', 'ok', 'status', 'detail']))
    short = ['label', 'status', 'startup_s', 'cases', 'answered', 'correct', 'requests', 'itl_b1_ms', 'itl_peak_b',
             'itl_peak_ms', 'preemptions', 'releases', 'readmissions', 'findings']
    md = ['# perf-mab tables (job %s, plan %s, %s)' % (result.get('run_label'), result.get('plan'), result.get('status')),
          '', 'Derived from the verified archive only. ITL: vLLM\'s inter-token histogram over clean /metrics intervals '
          '(the same running count at both ends, nothing waiting, no step above 0.5 s), by running count. Releases '
          '/ readmissions: capacity.jsonl (route B). Correct: synthetic code-word answers only.', '',
          md_table(servers, short), '', '## Cross-server comparisons', '',
          md_table(result.get('comparisons') or [], ['workload', 'base', 'other', 'cases', 'identical_turns',
                                                     'cases_fully_identical']), '',
          '## Replay (G1c)', '', 'turns %d, identical %d' % (len(replays), sum(x.get('identical') is True for x in replays)),
          '', '## Probes (G1d)', '', md_table(probes, ['server', 'name', 'ok', 'status']), '',
          '## Route B per session', '', md_table(sessions, ['server', 'session_index', 'releases', 'readmissions',
                                                             'released_pages', 're_releases'])]
    for label in sorted({s['server'] for s in sessions}):
        mine = [s for s in sessions if s['server'] == label]
        md += ['', "%s: %d releases, %d while the released session's next request waited for admission (sent, not yet "
               "readmitted; %d after the server certainly held it): an idle session in the runtime's sense, its "
               "readmission restored the Pages before that request's phase 1."
               % (label, sum(s['releases'] for s in mine), sum(s['queued_releases'] for s in mine),
                  sum(s['queued_at_server'] for s in mine))]
    (target / 'tables.md').write_text('\n'.join(md) + '\n', encoding='utf-8', newline='\n')
    written.append(target / 'tables.md')
    (target / 'servers.md').write_text(md_table(servers, fields) + '\n', encoding='utf-8', newline='\n')
    written.append(target / 'servers.md')
    return written


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    for path in write_tables(sys.argv[1]):
        print('TABLE %s' % path)
