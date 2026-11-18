"""perf-gemma-swe32 tables across jobs (local only): one row per server of every given join (join_gswe32.py output), in the
shape of the paper's Table 1, plus the per-instance resolution and turns matrices (gswe32: turns = model calls per
instance and policy, the question of the rerun), the prompt lengths against the context limit,
the per-turn admission / stall / selection numbers, the goal_query rewriter numbers and the failures.

Usage: python -B gswe32_tables.py --out <new dir> <join dir> [<join dir> ...]

Columns (as defined in the README; the throughput definition is PROVISIONAL):
accuracy = resolved / planned (32 tasks, as swe16: one task = 3.125 points, binomial s.e. ~7.7 points at p = 0.75);
throughput = client completion tokens / runner span (and / model span); cross-checked numerators. A server whose join
failed, whose evaluation is absent or whose run was partial is marked, never silently dropped.
"""
import argparse
import csv
import json
from pathlib import Path

TABLE = [('policy', 'Policy'), ('label', 'Server'), ('job', 'Job'), ('status', 'Run'), ('resolved', 'Resolved'),
         ('instances', 'Planned'), ('accuracy', 'Acc %'), ('throughput_runner', 'tok/s (runner span)'),
         ('throughput_model', 'tok/s (model span)'), ('span_runner_s', 'Span s'), ('completion_tokens', 'Gen tok (client)'),
         ('server_generated_tokens', 'Gen tok (server)'), ('vllm_generated_tokens', 'Gen tok (vLLM)'),
         ('calls', 'Calls'), ('turns_per_min', 'Turns/min'), ('admission_stall_fraction', 'Stall share'),
         ('queue_wait_p95', 'Queue wait p95 s'), ('capacity_waits', 'CAPACITY_WAIT'), ('selective_turns', 'Selective turns'),
         ('rebuilds', 'Rebuilds'), ('prompt_max', 'Max prompt tok'), ('context_refusals', 'Context refusals'),
         ('timeouts', 'Timeouts'), ('rewriter_calls', 'Rewriter calls'), ('rewrite_share', 'REWRITE share'),
         ('construct_ms_p50', 'Goal construct ms p50 (rewriter turns)'), ('fallbacks', 'Goal fallbacks'),
         ('joined', 'Joined'), ('kernels', 'Kernels (attention/FA/GDN prefill/sampler)'), ('flags', 'Flags')]


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
    rows, matrix, turns, prompts, failures = [], {}, {}, [], []
    for path in args.joins:
        value = load(Path(path) / 'join.json')
        job = value.get('run_id')
        for server in value['servers']:
            goal = server.get('goal') or {}
            flags = []
            if value.get('tasks_provisional'):
                flags.append('PROVISIONAL_TASKS')
            if value.get('provisional'):  # gswe32-config.json "provisional": the values not confirmed yet
                flags.append('PROVISIONAL(%s)' % ','.join(value['provisional']))
            if value.get('verdict') == 'JOIN_PASS_BULK_PENDING':
                flags.append('BULK_PENDING')
            elif value.get('verdict') != 'JOIN_PASS':
                flags.append('JOIN_FAIL')
            if server.get('evaluation') != 'present':
                flags.append('EVALUATION_ABSENT')
            if server.get('kernel_alignment') not in (None, 'PASS'):
                flags.append('KERNELS_%s' % server['kernel_alignment'])
            if server.get('done') != server.get('instances'):
                flags.append('PARTIAL')
            if not str(value.get('deltaai_verdict', '')).startswith(('GSWE32_PASS', 'ARCHIVE_VERIFIED')):
                flags.append('DELTAAI_%s' % value.get('deltaai_verdict'))
            rows.append(dict(server, job=job, status=value.get('deltaai_verdict'),
                             queue_wait_p95=(server.get('queue_wait_s') or {}).get('p95'),
                             prompt_max=(server.get('prompt_tokens') or {}).get('max'),
                             rewriter_calls=goal.get('rewriter_calls'), rewrite_share=goal.get('rewrite_share'),
                             construct_ms_p50=(goal.get('construct_ms_rewriter') or {}).get('p50'),
                             fallbacks=goal.get('fallbacks'), flags=' '.join(flags)))
            for bucket in server.get('prompt_buckets') or ():
                prompts.append(dict(job=job, label=server['label'], policy=server['policy'], **bucket))
        for record in value['instances']:
            matrix.setdefault(record['instance_id'], {})['%s/%s' % (record['policy'], record['label'])] = \
                {True: 'R', False: '-', None: '?'}.get(record.get('resolved'), '?')
            turns.setdefault(record['instance_id'], {})['%s/%s' % (record['policy'], record['label'])] = \
                record.get('calls')
        for failure in value.get('failed') or ():
            failures.append(dict(job=job, **{k: failure.get(k) for k in ('label', 'instance_id', 'check', 'detail')}))
    columns = sorted({k for v in matrix.values() for k in v})
    with (out / 'table1.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow([title for _, title in TABLE])
        writer.writerows([[row.get(key) for key, _ in TABLE] for row in rows])
    with (out / 'instances-matrix.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['instance'] + columns)
        writer.writerows([[i] + [matrix[i].get(c, '') for c in columns] for i in sorted(matrix)])
    with (out / 'turns-matrix.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['instance'] + columns)
        writer.writerows([[i] + [turns[i].get(c, '') for c in columns] for i in sorted(turns)])
    with (out / 'prompt-lengths.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['job', 'label', 'policy', 'low', 'high', 'turns'])
        writer.writerows([[p['job'], p['label'], p['policy'], p['low'], p['high'], p['turns']] for p in prompts])
    with (out / 'failures.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['job', 'label', 'instance_id', 'check', 'detail'])
        writer.writerows([[f['job'], f.get('label'), f.get('instance_id'), f.get('check'), str(f.get('detail'))[:300]]
                          for f in failures])
    provisional = sorted({p for path in args.joins for p in load(Path(path) / 'join.json').get('provisional') or ()})
    tasks_note = ('Task list PROVISIONAL in at least one join.' if any(
        load(Path(path) / 'join.json').get('tasks_provisional') for path in args.joins)
        else 'Task list: the first 32 of SWE-bench Verified (tasks/gswe32-tasks.json, as swe16).')
    ran = [r for r in rows if r.get('kernel_alignment') is not None]
    kernel_sets = sorted({r.get('kernels') or '?' for r in ran})
    alignment = ('KERNELS_ALIGNED' if ran and all(r['kernel_alignment'] == 'PASS' for r in ran) and len(kernel_sets) == 1
                 else 'KERNELS_MISALIGNED' if ran else 'KERNELS_UNKNOWN')
    text = ('# perf-gemma-swe32 results (one run per server)\n\n' + tasks_note + ' Still PROVISIONAL (gswe32-config.json '
            '"provisional"): ' + (', '.join(provisional) or 'nothing') + '; the throughput definition is runner span, '
            'client completion tokens.\n\nKernel alignment across the %d servers that ran: %s (%s).\n\n'
            % (len(ran), alignment, '; '.join(kernel_sets))
            + markdown(rows, TABLE) + '\n\n## Per instance (R resolved, - not, ? not evaluated)\n\n'
            + markdown([dict(instance=i, **matrix[i]) for i in sorted(matrix)], [('instance', 'Instance')]
                       + [(c, c) for c in columns]) + '\n\n## Turns per instance (model calls)\n\n'
            + markdown([dict(instance=i, **turns[i]) for i in sorted(turns)], [('instance', 'Instance')]
                       + [(c, c) for c in columns]) + '\n')
    (out / 'tables.md').write_bytes(text.encode('utf-8'))
    for row in rows:
        print('TABLE1 %s %s resolved=%s/%s acc=%s tok/s=%s span_s=%s flags=%s' % (
            row['policy'], row['label'], row.get('resolved'), row.get('instances'), row.get('accuracy'),
            row.get('throughput_runner'), row.get('span_runner_s'), row.get('flags')))
    print('%s servers=%d kernels=%s' % (alignment, len(ran), ';'.join(kernel_sets)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
