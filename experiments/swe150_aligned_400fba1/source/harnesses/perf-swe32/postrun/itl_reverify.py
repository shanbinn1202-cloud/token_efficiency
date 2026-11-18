"""Local only (never delivered): recompute the 2026-11-10 decode columns from a job's fetched evidence, old jobs too.

The job's sampler (measure.batch_profile in the payload) writes the `clean` cells and `decode_lost` for jobs built
after 2026-11-10; an older job's summary has the `steady` cells only. This reads one verified job's evidence (the
`evidence_dir` of its check\\local-verification.json: job-output\\<label>\\metrics\\samples.jsonl and the sample files,
each checked against its receipt's SHA-256 by measure.load_samples) and, per server with samples:
- reproduces the recorded batch profile's existing columns (stable / steady cells, stall_intervals, span, running
  mean) with this harness's measure.py: REPRODUCED or DIFFERENT (a different value would mean the old columns'
  definition changed; it must not);
- adds the clean cells (CLEAN_FACTOR x the median interval ITL at b, measure.clean_bound) next to the steady ones;
- prints the admission-pass share as recorded (`admission_stall_fraction` = sum of admission_engine_ms / span, the
  table's old "Stall share") and the decode time lost by vLLM's ITL histogram (measure.decode_lost: point and the
  review's bracket).
Counts and timings only. Writes <out>\\itl-reverify-<job>.json (new file) when --out is given.

Usage: python -I -B itl_reverify.py <verification folder or its check dir> [...] [--out DIR]
Exit 0 iff every server's existing columns were reproduced.
"""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import measure  # noqa: E402

OLD_TOP = ('samples', 'readable', 'span_s', 'running_peak', 'running_mean', 'waiting_peak', 'kv_usage_peak',
           'preemptions_max', 'stall_intervals')
OLD_CELL = ('batch', 'intervals', 'stable_s', 'stable_tok_s', 'steady_intervals', 'steady_s', 'steady_tok_s',
            'steady_itl_ms')


def existing_differ(recorded, profile):
    """The existing columns (before 2026-11-10) where a recomputed batch profile differs from the recorded one."""
    differ = [k for k in OLD_TOP if recorded.get(k) != profile.get(k)]
    old_cells = {c['batch']: c for c in recorded.get('batches') or ()}
    new_cells = {c['batch']: c for c in profile.get('batches') or ()}
    if sorted(old_cells) != sorted(new_cells):
        differ.append('batch sizes')
    return differ + ['b=%s %s' % (b, k) for b in sorted(old_cells) for k in OLD_CELL
                     if b in new_cells and old_cells[b].get(k) != new_cells[b].get(k)]


def check_dir_of(path):
    path = Path(path)
    return path / 'check' if (path / 'check' / 'local-verification.json').is_file() else path


def reverify(path):
    check = check_dir_of(path)
    report = json.loads((check / 'local-verification.json').read_bytes())
    files = Path(report['evidence_dir']) / 'files'
    value = dict(job_id=report.get('job_id'), run_id=report.get('run_id'), verdict=report.get('verdict'),
                 evidence_dir=str(report['evidence_dir']), servers={},
                 clean_rule=dict(factor=measure.CLEAN_FACTOR, cap_s=measure.CLEAN_CAP_SECONDS))
    for out in sorted(p for p in (files / 'job-output').iterdir() if (p / 'metrics' / 'samples.jsonl').is_file()):
        label = out.name
        summary = json.loads((files / ('summary-%s.json' % label)).read_bytes())
        measured = summary.get('measurements') or {}
        recorded = measured.get('batch') or {}
        profile = measure.batch_profile(measure.batch_timeline(out))
        differ = existing_differ(recorded, profile)
        cells =[dict({k: c.get(k) for k in OLD_CELL}, median_interval_itl_ms=c['median_interval_itl_ms'],
                      clean_bound_s=c['clean_bound_s'], clean_intervals=c['clean_intervals'], clean_s=c['clean_s'],
                      clean_tok_s=c['clean_tok_s'], clean_itl_ms=c['clean_itl_ms'],
                      steady_over_clean=round(c['steady_itl_ms'] / c['clean_itl_ms'] - 1, 4)
                      if c.get('steady_itl_ms') and c.get('clean_itl_ms') else None) for c in profile['batches']]
        value['servers'][label] = dict(
            policy=summary.get('policy'), kind=summary.get('kind'),
            existing_columns='REPRODUCED' if not differ else 'DIFFERENT', differ=differ[:20],
            admission_pass_share=measured.get('admission_stall_fraction'),
            admission_pass_s=measured.get('admission_stall_s'), span_s=measured.get('span_s'),
            stall_intervals=profile['stall_intervals'], decode_lost=profile['decode_lost'], cells=cells)
    return value


def show(value):
    print('JOB %s %s %s' % (value['job_id'], value['run_id'], value['verdict']))
    for label, s in value['servers'].items():
        lost = s['decode_lost'] or {}
        print('SERVER %s %s existing_columns=%s %s' % (label, s['policy'], s['existing_columns'],
                                                         ' '.join(s['differ'][:5])))
        print('  admission-pass share (recorded "Stall share") %s = %s s / span %s s' % (
            s['admission_pass_share'], s['admission_pass_s'], s['span_s']))
        if lost:
            print('  decode time lost (ITL histogram) %.1f %% [%.1f-%.1f %%] = %.0f s [%.0f-%.0f] of ITL sum %.0f s '
                  '(%d observations, mean ITL %.2f ms, largest clean step %.2f ms)' % (
                      100 * lost['lost_share'], 100 * lost['lost_share_low'], 100 * lost['lost_share_high'],
                      lost['lost_s'], lost['lost_low_s'], lost['lost_high_s'], lost['itl_sum_s'],
                      lost['observations'], lost['mean_itl_ms'], lost['largest_clean_itl_ms']))
        print('  b   stable steady steady_itl_ms | bound_s clean clean_itl_ms steady/clean-1')
        for c in s['cells']:
            print('  %-3d %-6d %-6d %-13s | %-7s %-5d %-12s %s' % (
                c['batch'], c['intervals'], c['steady_intervals'], c['steady_itl_ms'], c['clean_bound_s'],
                c['clean_intervals'], c['clean_itl_ms'],
                '%+.1f%%' % (100 * c['steady_over_clean']) if c['steady_over_clean'] is not None else '-'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folders', nargs='+')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    bad = 0
    for folder in args.folders:
        value = reverify(folder)
        show(value)
        bad += sum(1 for s in value['servers'].values() if s['existing_columns'] != 'REPRODUCED')
        if args.out:
            path = Path(args.out) / ('itl-reverify-%s.json' % value['job_id'])
            Path(args.out).mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8', newline='\n') as stream:
                json.dump(value, stream, sort_keys=True, indent=1)
                stream.write('\n')
            print('WROTE %s' % path)
    print('ITL_REVERIFY_%s servers_not_reproduced=%d' % ('PASS' if not bad else 'FAIL', bad))
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
