"""The clean ITL cells and the decode time lost (2026-11-10, the perf review's FINDINGS 1.1 / 1.2) recomputed from a
fetched job's evidence, old jobs included (local only; read-only on the evidence; no re-verification).

A job verified before 2026-11-10 recorded only the `steady` cells (measure.STALL_SECONDS: an interval is steady
unless it holds an ITL above 0.5 s, so today's 0.2-0.4-s admissions stay inside them). This tool reads the job's
/metrics samples from the verified evidence directory (the check dir's local-verification.json `evidence_dir`) and
runs this harness's measure.batch_profile on them: per server and batch size b the steady cells (they must equal the
job's recorded summary: the old columns are reproduced), the clean cells (measure.CLEAN_RULE) and the decode time the
requests lost (measure.itl_excess: sum of ITL - observations x the clean step at the interval's batch size, a share of
the ITL sum, low-high). Nothing in the evidence or the verification changes.

Usage: python -B itl_columns.py <check dir> [--out <new dir>]   (writes itl.json and itl.md; prints one line per server)
Exit 0 when every server's recomputed steady cells equal the recorded ones, else 1.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]
import measure  # noqa: E402

CELL_KEYS = ('batch', 'intervals', 'stable_s', 'stable_tok_s', 'steady_intervals', 'steady_s', 'steady_tok_s',
             'steady_itl_ms')


def servers(evidence):
    """(label, job-output dir, recorded summary or None) of every server with /metrics samples."""
    root = evidence / 'job-output'
    for out in sorted(p for p in root.iterdir() if p.is_dir() and (p / measure.METRICS_DIR / 'samples.jsonl').is_file()):
        path = evidence / ('summary-' + out.name + '.json')
        yield out.name, out, json.loads(path.read_bytes()) if path.is_file() else None


def recompute(check_dir):
    report = json.loads((Path(check_dir) / 'local-verification.json').read_bytes())
    evidence = Path(report['evidence_dir']) / 'files'
    result = dict(check_dir=str(check_dir), evidence_dir=str(evidence), job_verdict=report.get('verdict'),
                  run_id=report.get('run_id'), clean_rule=measure.CLEAN_RULE, servers=[])
    for label, out, summary in servers(evidence):
        timeline, error = measure.load_timeline(out)
        profile = measure.batch_profile(timeline)
        recorded = ((summary or {}).get('measurements') or {}).get('batch')
        same = None
        if recorded is not None:
            old = [{k: row.get(k) for k in CELL_KEYS} for row in recorded.get('batches') or ()]
            new = [{k: row.get(k) for k in CELL_KEYS} for row in profile['batches']]
            same = old == new and recorded.get('stall_intervals') == profile['stall_intervals']
        result['servers'].append(dict(label=label, timeline_error=error, steady_equal_recorded=same,
                                      span_s=profile['span_s'], running_mean=profile['running_mean'],
                                      stall_intervals=profile['stall_intervals'], itl_excess=profile['itl_excess'],
                                      batches=profile['batches']))
    return result


def pct(seconds, total):
    """A share in per cent from the seconds (not the 4-decimal share: no double rounding)."""
    return '' if seconds is None or not total else '%.1f' % (100 * seconds / total)


def markdown(result):
    lines = ['# Clean ITL cells and the decode time lost (%s)' % result['run_id'], '',
             'Evidence `%s` (job verdict %s). Rule: %s. The steady cells are the job\'s recorded ones (0.5-s rule, '
             'today\'s admissions inside); "lost" = sum of ITL - observations x the clean step at the interval\'s '
             'batch size, as a share of the ITL sum (low-high: the longer / shorter of the interval ends\' clean '
             'steps); request-seconds.' % (result['evidence_dir'], result['job_verdict'], result['clean_rule']), '',
             '| Server | Steady = recorded | ITL sum req-s | Observations | Decode lost req-s (low-high) | '
             'Decode lost % of ITL sum (low-high) |', '| --- | --- | ---: | ---: | --- | --- |']
    for s in result['servers']:
        x = s['itl_excess'] or {}
        lines.append('| %s | %s | %s | %s | %s-%s | %s-%s |' % (
            s['label'], s['steady_equal_recorded'], x.get('itl_sum_s'), x.get('observations'), x.get('excess_s_low'),
            x.get('excess_s_high'), pct(x.get('excess_s_low'), x.get('itl_sum_s')),
            pct(x.get('excess_s_high'), x.get('itl_sum_s'))))
    for s in result['servers']:
        lines += ['', '## %s: per batch size' % s['label'], '',
                  '| b | Stable intervals | Steady intervals | Steady ITL ms (0.5-s rule) | Clean bound ms | '
                  'Clean intervals | Clean ITL ms | Steady / clean | Clean tok/s |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for b in s['batches']:
            ratio = '%+.1f %%' % (100 * (b['steady_itl_ms'] / b['clean_itl_ms'] - 1)) \
                if b.get('steady_itl_ms') and b.get('clean_itl_ms') else ''
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                b['batch'], b['intervals'], b['steady_intervals'], b['steady_itl_ms'], b['clean_bound_ms'],
                b['clean_intervals'], b['clean_itl_ms'], ratio, b['clean_tok_s']))
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('check_dir')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    result = recompute(args.check_dir)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True)
        (out / 'itl.json').write_text(json.dumps(result, indent=1, sort_keys=True) + '\n', encoding='utf-8')
        (out / 'itl.md').write_bytes(markdown(result).encode('utf-8'))
    for s in result['servers']:
        x = s['itl_excess'] or {}
        print('ITL_COLUMNS %s steady_equal_recorded=%s itl_sum_s=%s observations=%s lost_s=%s-%s lost_share=%s-%s' % (
            s['label'], s['steady_equal_recorded'], x.get('itl_sum_s'), x.get('observations'), x.get('excess_s_low'),
            x.get('excess_s_high'), x.get('share_low'), x.get('share_high')))
    ok = all(s['steady_equal_recorded'] is not False for s in result['servers'])
    print('ITL_COLUMNS_%s servers=%d' % ('PASS' if ok else 'FAIL', len(result['servers'])))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
