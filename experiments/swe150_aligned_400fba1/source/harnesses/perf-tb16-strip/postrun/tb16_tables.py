"""perf-tb16 tables across jobs (local only): one row per server of every given join (join_tb16.py output), in the
shape of the paper's Table 1 (the eval pipeline's column names: 'Policy', 'Server', 'Resolved', 'Planned',
'Gen tok (client)', 'Span s', 'tok/s (runner span)'; for Terminal-Bench the "runner span" is the agents' span: first
agent start to last agent end), plus the per-task reward matrix across policies and the per-task call counts.

Usage: python -B tb16_tables.py --out <new dir> <join dir> [<join dir> ...]

Accuracy = solved (reward 1.0) / planned (89 tasks: one task = 1.12 points, binomial s.e. ~5.3 points at p = 0.5);
throughput = ATIF main-agent completion tokens / the agents' span (PROVISIONAL definition); cross-checked numerators.
A server whose join failed or whose run was partial is marked, never silently dropped.
"""
import argparse
import csv
import json
from pathlib import Path

TABLE = [('policy', 'Policy'), ('label', 'Server'), ('job', 'Job'), ('status', 'Run'), ('resolved', 'Resolved'),
         ('instances', 'Planned'), ('accuracy', 'Acc %'), ('throughput_tokens_per_s', 'tok/s (runner span)'),
         ('span_agent_s', 'Span s'), ('completion_tokens', 'Gen tok (client)'),
         ('summary_completion_tokens', 'Summary gen tok (client)'), ('server_generated_tokens', 'Gen tok (server)'),
         ('vllm_generated_tokens', 'Gen tok (vLLM)'), ('job_wall_s', 'Harbor job wall s'), ('calls', 'Calls'),
         ('turns_per_min', 'Turns/min'), ('queue_wait_p95', 'Queue wait p95 s'), ('open_wait_p95', 'Open wait p95 s'),
         ('rebuilds', 'Rebuilds'), ('rebinds', 'Rebinds'), ('selective_turns', 'Selective turns'),
         ('prompt_max', 'Max prompt tok'), ('context_refusals', 'Context refusals'),
         ('summarizations', 'Summarizations'), ('timeouts', 'Agent timeouts'),
         ('unrecorded_completion_tokens', 'Unrecorded tok (timeout-cut)'), ('short_summary_calls', 'Short summary calls'),
         ('short_summary_completion_tokens', 'Short summary tok (unrecorded)'), ('no_call_steps', 'Fallback steps (no call)'),
         ('length_cut_calls', 'Length-cut replies (unrecorded)'),
         ('length_cut_completion_tokens', 'Length-cut tok (unrecorded)'),
         ('cancelled_requests', 'Cancelled requests'),
         ('cancelled_generated_tokens', 'Cancelled gen tok (vLLM - server)'), ('joined', 'Joined'),
         ('kernels', 'Kernels (attention/FA/GDN prefill/sampler)'), ('flags', 'Flags'),
         # 2026-11-10 (the perf review, FINDINGS 1.1 / 1.2): join_tb16's decode-lost figure (a join made before it:
         # empty; rerun the join, or postrun/itl_columns.py on the check dir)
         ('decode_lost_pct', 'Decode lost % of ITL sum (clean steps, low-high)'),
         # 2026-11-15 (job 3560498): join_tb16's killed in-flight rows (a join made before it: empty)
         ('killed_in_flight_calls', 'Killed in-flight requests'),
         ('killed_in_flight_completion_tokens', 'Killed in-flight tok (never received)')]


def load(path):
    return json.loads(Path(path).read_bytes())


def markdown(rows, columns):
    lines = ['| ' + ' | '.join(title for _, title in columns) + ' |', '|' + ' --- |' * len(columns)]
    lines += ['| ' + ' | '.join('' if row.get(key) is None else str(row.get(key)) for key, _ in columns) + ' |'
              for row in rows]
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('joins', nargs='+')
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True)
    rows, matrix, calls, policies = [], {}, {}, []
    for folder in args.joins:
        value = load(Path(folder) / 'join.json')
        job = str(value.get('run_id') or '').split('-')[1:2]
        for server in value['servers']:
            flags = []
            if value['verdict'] != 'JOIN_PASS':
                flags.append(value['verdict'])
            if server.get('done') is not None and server.get('done') != server.get('instances'):
                flags.append('PARTIAL %s/%s done' % (server.get('done'), server.get('instances')))
            if server.get('note'):
                flags.append(server['note'])
            row = dict(server, job=job[0] if job else None, status=value.get('deltaai_verdict'), flags='; '.join(flags),
                       queue_wait_p95=(server.get('queue_wait_s') or {}).get('p95'),
                       open_wait_p95=(server.get('open_wait_s') or {}).get('p95'),
                       prompt_max=(server.get('prompt_tokens') or {}).get('max'))
            rows.append(row)
            key = '%s/%s' % (server['policy'], server['label'])
            policies.append(key)
            for record in value['instances']:
                if record['label'] == server['label']:
                    matrix.setdefault(record['instance_id'], {})[key] = record.get('reward')
                    calls.setdefault(record['instance_id'], {})[key] = record.get('calls')
    with (out / 'table1.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow([title for _, title in TABLE])
        writer.writerows([[row.get(k) for k, _ in TABLE] for row in rows])
    for name, values in (('rewards.csv', matrix), ('calls.csv', calls)):
        with (out / name).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['Task'] + policies)
            for task in sorted(values):
                writer.writerow([task] + [values[task].get(p) for p in policies])
    text = ('# perf-tb16 tables\n\n' + markdown(rows, TABLE) + '\n\nAccuracy = solved / planned; throughput = ATIF '
            'main-agent completion tokens / agents\' span (PROVISIONAL; the column keeps the eval pipeline\'s name '
            '"runner span": here the first agent start to the last agent end). Decode lost: the decode time the '
            'requests lost to steps longer than the plain step of their batch size, % of vLLM\'s ITL sum (low-high). '
            'Unrecorded, short-summary, length-cut and killed in-flight cells are empty where the server\'s rows were '
            'not read (the stock control, no bulk archive). Killed in-flight: the request in flight when the client '
            'killed its trial (agent timeout), answered on the server, never received (join.json findings). rewards.csv: reward per task and server; calls.csv: model calls per '
            'task and server.\n')
    (out / 'tables.md').write_bytes(text.encode('utf-8'))
    print('TABLE1 %d rows -> %s' % (len(rows), out / 'table1.csv'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
