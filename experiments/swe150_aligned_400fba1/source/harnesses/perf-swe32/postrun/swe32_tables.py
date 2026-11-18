"""perf-swe32 tables across jobs (local only): one row per server of every given join (join_swe32.py output), in the
shape of the paper's Table 1, plus the per-instance resolution and turns matrices (swe32: turns = model calls per
instance and policy, the question of the rerun), the prompt lengths against the context limit,
the per-turn admission / stall / selection numbers, the goal_query rewriter numbers and the failures.

Usage: python -B swe32_tables.py --out <new dir> <join dir> [<join dir> ...]

Columns (as defined in the README; the throughput definition is PROVISIONAL):
accuracy = resolved / planned (32 tasks, as swe16: one task = 3.125 points, binomial s.e. ~7.7 points at p = 0.75);
throughput = client completion tokens / runner span (and / model span); cross-checked numerators. A server whose join
failed, whose evaluation is absent or whose run was partial is marked, never silently dropped.
2026-11-10 (perf review FINDINGS 1.1, 1.2, 2.8, section 3; the old columns keep their values, titles that said
something else than the number were changed; the eval pipeline reads 'Planned', 'Resolved', 'Gen tok (client)',
'Span s', 'tok/s (runner span)' and 'Selective turns', which keep their titles and values):
- 'Admission-pass share (engine time / span)' (was "Stall share"): sum of admission_engine_ms / span; it times the
  admission pass only, not what the decoding requests lose;
- 'Decode time lost % of ITL sum' [low, high]: vLLM's ITL sum minus its observations x the clean step at the
  interval's batch size (measure.decode_lost; the bracket: every observation at the largest clean step / each interval
  at the faster end);
- 'tok/s (runner span; server tokens for instances without a trajectory)': a cut instance's server-generated tokens
  added ('tok/s (runner span)' = 'Gen tok (client)' / 'Span s', client tokens only, as before);
- 'Goal-gate REWRITE decisions / turns' (was "REWRITE share"): turns whose Goal gate decided REWRITE / turns, applied
  or not (on observation_query the gate decides and no rewriter runs: job 3461703's obs262 shows 0.0646 with 0
  rewriter calls); 'Server queue wait p95 s (FIFO + CAPACITY_WAIT retries)' (was "Queue wait p95 s"): the runtime
  row's queue wait; empty on the stock control;
- batch-itl.csv: per server and batch size the steady ITL (no ITL above 0.5 s: holds today's admissions) and the clean
  ITL (the plain decode step, measure.CLEAN_FACTOR) with their intervals.
2026-11-12 (the 64-instance proxy): 'Goal wait share of session wall' = the sum of the completed requests' goal_async_ms
(the time a request's admission waited for its Goal: parked from the Goal's start to its resume; with --goal-overlap only
what phase 1 did not hide) / the sum of the instances' runner walls ('Wall s', the instances' job completion times);
'Goal wait s (sum)' its numerator, 'Goal wait ms p50 (waiting turns)' over the requests that waited (join_swe32.goal_wait;
goal_query servers only, empty elsewhere). A join of the 64-task list says so in the task note.
"""
import argparse
import csv
import json
from pathlib import Path

TABLE = [('policy', 'Policy'), ('label', 'Server'), ('job', 'Job'), ('status', 'Run'), ('resolved', 'Resolved'),
         ('instances', 'Planned'), ('accuracy', 'Acc %'), ('throughput_runner', 'tok/s (runner span)'),
         ('throughput_model', 'tok/s (model span)'), ('span_runner_s', 'Span s'), ('completion_tokens', 'Gen tok (client)'),
         ('no_trajectory', 'Started without trajectory'), ('tokens_all', 'Gen tok (client; server for those)'),
         ('throughput_all', 'tok/s (runner span; server tokens for instances without a trajectory)'),
         ('server_generated_tokens', 'Gen tok (server)'), ('vllm_generated_tokens', 'Gen tok (vLLM)'),
         ('calls', 'Calls'), ('turns_per_min', 'Turns/min'),
         # 2026-11-16: the steady state (K sessions in flight) and the tail, next to the whole run
         # (join_swe32.STEADY_NOTE)
         ('steady_tok_s', 'tok/s (steady, K in flight)'), ('steady_turns_per_min', 'Turns/min (steady)'),
         ('steady_s', 'Steady s'), ('tail_tok_s', 'tok/s (tail)'), ('tail_turns_per_min', 'Turns/min (tail)'),
         ('tail_s', 'Tail s'), ('rollbacks', 'Rollbacks'), ('thinking_rebuilds', 'Rebuilds (thinking history)'),
         ('admission_stall_fraction', 'Admission-pass share (engine time / span)'),
         ('decode_lost_pct', 'Decode time lost % of ITL sum'), ('decode_lost_pct_low', 'Decode lost % low'),
         ('decode_lost_pct_high', 'Decode lost % high'),
         ('queue_wait_p95', 'Server queue wait p95 s (FIFO + CAPACITY_WAIT retries)'), ('capacity_waits', 'CAPACITY_WAIT'),
         ('selective_turns', 'Selective turns'),
         ('rebuilds', 'Rebuilds'), ('prompt_max', 'Max prompt tok'), ('context_refusals', 'Context refusals'),
         ('timeouts', 'Timeouts'), ('rewriter_calls', 'Rewriter calls'), ('rewrite_share', 'Goal-gate REWRITE decisions / turns'),
         ('construct_ms_p50', 'Goal construct ms p50 (rewriter turns)'),
         # 2026-11-12 (README "Since 2026-11-12"): the Goal wait on the sessions' critical path (join_swe32.goal_wait)
         ('goal_wait_share', 'Goal wait share of session wall'), ('goal_wait_s', 'Goal wait s (sum)'),
         ('goal_wait_ms_p50', 'Goal wait ms p50 (waiting turns)'), ('fallbacks', 'Goal fallbacks'),
         ('joined', 'Joined'), ('kernels', 'Kernels (attention/FA/GDN prefill/sampler)'), ('flags', 'Flags')]


# per server and batch size b (join row 'batch': the job's profile, or recomputed for a job before 2026-11-10)
BATCH_ITL = [('job', 'Job'), ('label', 'Server'), ('policy', 'Policy'), ('source', 'Profile'), ('batch', 'Batch'),
             ('intervals', 'Stable intervals'), ('steady_intervals', 'Steady intervals (no ITL > 0.5 s)'),
             ('steady_itl_ms', 'Steady ITL ms (holds admissions)'), ('steady_tok_s', 'Steady tok/s'),
             ('median_interval_itl_ms', 'Median interval ITL ms'), ('clean_bound_s', 'Clean bound s'),
             ('clean_intervals', 'Clean intervals'), ('clean_itl_ms', 'Clean ITL ms (plain step)'),
             ('clean_tok_s', 'Clean tok/s')]


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
    rows, matrix, turns, prompts, failures, cells = [], {}, {}, [], [], []
    for path in args.joins:
        value = load(Path(path) / 'join.json')
        job = value.get('run_id')
        for server in value['servers']:
            goal = server.get('goal') or {}
            flags = []
            if value.get('tasks_provisional'):
                flags.append('PROVISIONAL_TASKS')
            if value.get('provisional'):  # swe32-config.json "provisional": the values not confirmed yet
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
            if not str(value.get('deltaai_verdict', '')).startswith(('SWE32_PASS', 'ARCHIVE_VERIFIED')):
                flags.append('DELTAAI_%s' % value.get('deltaai_verdict'))
            rows.append(dict(server, job=job, status=value.get('deltaai_verdict'),
                             queue_wait_p95=(server.get('queue_wait_s') or {}).get('p95'),
                             prompt_max=(server.get('prompt_tokens') or {}).get('max'),
                             rewriter_calls=goal.get('rewriter_calls'), rewrite_share=goal.get('rewrite_share'),
                             construct_ms_p50=(goal.get('construct_ms_rewriter') or {}).get('p50'),
                             fallbacks=goal.get('fallbacks'), flags=' '.join(flags)))
            for bucket in server.get('prompt_buckets') or ():
                prompts.append(dict(job=job, label=server['label'], policy=server['policy'], **bucket))
            for cell in (server.get('batch') or {}).get('batches') or ():
                cells.append(dict(job=job, label=server['label'], policy=server['policy'],
                                  source=server.get('batch_source'), **{k: cell.get(k) for k, _ in BATCH_ITL[4:]}))
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
    with (out / 'batch-itl.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow([title for _, title in BATCH_ITL])
        writer.writerows([[c.get(key) for key, _ in BATCH_ITL] for c in cells])
    with (out / 'failures.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['job', 'label', 'instance_id', 'check', 'detail'])
        writer.writerows([[f['job'], f.get('label'), f.get('instance_id'), f.get('check'), str(f.get('detail'))[:300]]
                          for f in failures])
    provisional = sorted({p for path in args.joins for p in load(Path(path) / 'join.json').get('provisional') or ()})
    counts = sorted({load(Path(path) / 'join.json').get('task_count') or 32 for path in args.joins})
    tasks_note = ('Task list PROVISIONAL in at least one join.' if any(
        load(Path(path) / 'join.json').get('tasks_provisional') for path in args.joins)
        else 'Task list: the first 32 of SWE-bench Verified (tasks/swe32-tasks.json, as swe16).' if counts == [32] else
        'Task lists: the first %s of SWE-bench Verified in dataset order (tasks/swe32-tasks.json; 64: '
        'tasks/swe64-tasks.json, its first 32 the 32 list; 150: tasks/swe150-tasks.json, its first 64 the 64 list); '
        'rows of different task counts are not comparable.'
        % ' / '.join(str(n) for n in counts))
    ran = [r for r in rows if r.get('kernel_alignment') is not None]
    kernel_sets = sorted({r.get('kernels') or '?' for r in ran})
    alignment = ('KERNELS_ALIGNED' if ran and all(r['kernel_alignment'] == 'PASS' for r in ran) and len(kernel_sets) == 1
                 else 'KERNELS_MISALIGNED' if ran else 'KERNELS_UNKNOWN')
    text = ('# perf-swe32 results (one run per server)\n\n' + tasks_note + ' Still PROVISIONAL (swe32-config.json '
            '"provisional"): ' + (', '.join(provisional) or 'nothing') + '; the throughput definition is runner span, '
            'client completion tokens.\n\nColumns (2026-11-10): "tok/s (runner span)" = client completion tokens / '
            'runner span (first runner start to last runner end, stretches without a model call included); an '
            'instance cut at the stop without a trajectory has no client tokens there, "tok/s (runner span; server '
            'tokens for instances without a trajectory)" adds its server-generated tokens (empty for the stock '
            'control). "Admission-pass share" = sum of the admission passes\' engine time / span (the table\'s former '
            '"Stall share"); "Decode time lost %% of ITL sum" [low, high] = vLLM\'s ITL sum minus its observations x '
            'the clean step at the batch size, the decode time the running requests lost to everything that is not a '
            'plain step. batch-itl.csv: steady (no ITL > 0.5 s) and clean (the plain step) ITL per batch size.\n\n'
            'Steady state (2026-11-16): "tok/s (steady, K in flight)" / "Turns/min (steady)" over the '
            'runner-timeline stretches with exactly K sessions in flight (K = the server\'s '
            'max_running_sessions, so at least K tasks remain), from the first moment the pool fills to the '
            'last moment it is full; "tail" is the runner span after it; client completion tokens of the calls '
            'whose response time falls inside.\n\n'
            'Kernel alignment across the %d servers that ran: %s (%s).\n\n'
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
