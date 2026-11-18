"""Unit tests of postrun/run150_tables.py (the 150-run tables: stock control vs full vs ledger_query, NOT AUDITED),
on synthetic verified job directories (join.json, check file, table1.csv, ledger.json, RESULT.json) and a tiny core
evidence archive (tar.xz) built in a temporary directory (localtemp: on D:, never C:).

Run: python -B -m unittest test_run150_tables -v
"""
import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'postrun')]
import localtemp  # noqa: E402

localtemp.ensure()
import run150_tables as rt  # noqa: E402

STAMP = '20261120T000000'


def instance_records(label, policy, n, K):
    """n instances, K runners: instance k starts when a runner frees up (start times staggered)."""
    records = []
    for k in range(n):
        start = 1000.0 + 50.0 * (k // K) + k
        wall = 100.0 + 25.0 * k
        calls = 4 + k
        completion = 60 * calls
        records.append(dict(
            instance_id='repo__repo-%03d' % k, label=label, policy=policy, session=k, status='done',
            exit_status='Submitted', start=start, end=start + wall, wall_s=wall, calls=calls,
            completion_tokens=completion, prompt_tokens_total=1000 * calls, format_errors=k % 2,
            resolved=k % 3 != 1, eval_status='resolved' if k % 3 != 1 else 'unresolved',
            call_tokens=[[start + 1 + j, 60] for j in range(calls)], join='PASS'))
    return records


def make_job(root, run_id, plan, label, policy, n=6, K=4, release=False, ledger=False, core=True, thinking=True,
             result=True, preempted=5):
    job_id = run_id.split('-')[1]
    run_dir = Path(root) / 'out' / run_id
    job_dir = run_dir / ('swe32-job%s-%s' % (job_id, STAMP))
    stock = policy == 'stock'
    records = instance_records(label, policy, n, K)
    starts, ends = [r['start'] for r in records], [r['end'] for r in records]
    span = round(max(ends) - min(starts), 3)
    calls = sum(r['calls'] for r in records)
    completion = sum(r['completion_tokens'] for r in records)
    rollbacks = sum(r['format_errors'] for r in records)
    turns, checks = [], []
    for r in records:
        for j in range(r['calls']):
            waits = 2 if (r['session'] % 2 == 1 and j == 0) else 0
            turns.append(dict(session=r['session'], instance_id=r['instance_id'], capacity_waits=waits,
                              queue_wait_s=0.5, prompt_tokens=1000 + 10 * j, generated_tokens=60,
                              selection=dict(reason='cold_start' if j == 0 else None, active=j > 0, selected=8,
                                             candidate_pages=16)))
        for j in range(r['format_errors']):
            checks.append(dict(check='s%d-turn-%d-rollback' % (r['session'], 2 + j), status='PASS', detail={}))
    capacity = [dict(event='evict', session_index=0, pages=10, request_id='a'),
                dict(event='readmit', session_index=0, pages=10, request_id='b'),
                dict(event='evict', session_index=1, pages=7, request_id='c'),
                dict(event='evict', session_index=0, pages=4, request_id='d')]
    totals = dict(evictions=3, evicted_pages=21, readmissions=1, readmitted_pages=10)
    server = dict(label=label, policy=policy, evaluation='present', instances=n, done=n,
                  resolved=sum(1 for r in records if r['resolved']), calls=calls, completion_tokens=completion,
                  span_runner_s=span, throughput_runner=round(completion / span, 2),
                  turns_per_min=round(calls / span * 60, 2), running=K,
                  steady=dict(seconds=40.0, calls=20, tok_s=30.0, tokens=1200))
    if not stock:
        server.update(capacity_waits=sum(t['capacity_waits'] for t in turns),
                      selective_turns=sum(1 for t in turns if t['selection']['reason'] is None))
        if thinking:
            server.update(rollbacks=rollbacks, thinking_rebuilds=n)
        if release:
            server.update(idle_evictions=3, idle_readmissions=1, idle_readmitted_pages=10)
    join = dict(run_id=run_id, plan=plan, deltaai_verdict='SWE32_PASS_VERIFIED', verdict='JOIN_PASS',
                servers=[server], instances=records,
                keep_render={} if stock else {label: dict(calls=calls, exact_equal=calls, failed=0, status='DONE')})
    (job_dir / ('join-done-' + STAMP)).mkdir(parents=True)
    (job_dir / ('join-done-' + STAMP) / 'join.json').write_text(json.dumps(join), encoding='utf-8')
    (job_dir / 'check').mkdir()
    if stock:
        check = dict(policy='stock', checks=[dict(
            check='no-preemption', status='FINDING' if preempted else 'PASS', finding='STOCK_PREEMPTED',
            detail="vLLM preempted: [('metrics/sample-3.prom', 2.0), ('metrics/sample-9.prom', %d.0)]" % preempted)])
    else:
        histories = dict(append=calls - n, fresh=n, rebuild=n, **(dict(rollback=rollbacks) if thinking else {}))
        thinking_detail = dict(histories=histories, thinking=True, **(dict(rollbacks=rollbacks) if thinking else {}))
        release_detail = dict(totals, release_idle_pages=True, dropped_by_rebuild=1, pending_at_end=1) if release \
            else dict(release_idle_pages=None)
        check = dict(policy=policy, turns=turns, active_selection_turns=server['selective_turns'], checks=checks + [
            dict(check='thinking', status='PASS', detail=thinking_detail),
            dict(check='release-idle-pages', status='PASS', detail=release_detail),
            dict(check='session-pool', status='PASS', detail={}),
            dict(check='capacity-stall', status='PASS', detail=dict(stalls=0, limit_seconds=300))])
    (job_dir / 'check' / ('checks-%s.json' % label)).write_text(json.dumps(check), encoding='utf-8')
    (job_dir / ('tables-done-' + STAMP)).mkdir()
    with (job_dir / ('tables-done-' + STAMP) / 'table1.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['Policy', 'Server', 'Resolved', 'Calls', 'Span s', 'Gen tok (client)', 'tok/s (runner span)'])
        writer.writerow([policy, label, server['resolved'], calls, span, completion, server['throughput_runner']])
    if ledger:
        (job_dir / ('ledger-done-' + STAMP)).mkdir()
        (job_dir / ('ledger-done-' + STAMP) / 'ledger.json').write_text(json.dumps(dict(
            status='PASS', servers={label: dict(policy=policy, parity=dict(status='PASS', stats=dict(
                requests=server['selective_turns'], equal=server['selective_turns'], differs=0, inactive=n)),
                endpoints=dict(resolved=server['resolved'], steps_per_instance=round(calls / n, 1), hook=dict(
                    requests=server['selective_turns'], ledger_pages_mean=12.5, items=dict(attended=3, skipped=1)),
                    selection_ms=dict(p50=70.0, p95=150.0)))})), encoding='utf-8')
    if result:
        (run_dir / 'RESULT.json').write_text(json.dumps(dict(
            run_id=run_id, job_id=job_id, job_dir='/elsewhere/out/%s/%s' % (run_id, job_dir.name),
            verdict='SWE32_PASS_VERIFIED', join_dir='join-done-' + STAMP, tables_dir='tables-done-' + STAMP,
            ledger_dir=('ledger-done-' + STAMP) if ledger else None)), encoding='utf-8')
    if core:
        evidence = Path(root) / 'evidence' / run_id
        evidence.mkdir(parents=True)
        final = (preempted + 2) if stock else 0  # the check quotes only the first scrapes: the last sample has more
        prom = lambda p, s: ('# HELP x\nvllm:num_preemptions_total{engine="0",model_name="m"} %d.0\n'  # noqa: E731
                             'vllm:request_success_total{engine="0",finished_reason="stop"} %d.0\n'
                             'vllm:request_success_total{engine="0",finished_reason="length"} 1.0\n' % (p, s)).encode()
        members = {'S0.json': b'{"x": 1}', 'job-output/%s/metrics/sample-0.prom' % label: prom(0, 3),
                   'job-output/%s/metrics/sample-10.prom' % label: prom(final, 3 + calls),
                   'job-output/%s/metrics/sample-9.prom' % label: prom(0, 3)}
        if not stock:
            members.update({'job-output/%s/runtime.jsonl' % label: b'{}\n' * 10,
                            'job-output/%s/engine-cleanup.json' % label: json.dumps(
                                dict(idle_release=totals) if release else {}).encode()})
        if release:
            members['job-output/%s/capacity.jsonl' % label] = ''.join(
                json.dumps(e) + '\n' for e in capacity).encode()
        with tarfile.open(str(evidence / 'swe32-core.tar.xz'), 'w:xz') as tar:
            for name, data in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return run_dir, job_dir


def run(argv):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        code = rt.main([str(a) for a in argv])
    return code, stream.getvalue()


class Run150Tables(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='run150-tables-'))
        self.addCleanup(shutil.rmtree, str(self.root), True)

    def three(self, run_ids=('swe32-9000001-r1', 'swe32-9000002-r1', 'swe32-9000003-r2'),
              plans=('stockkeep', 'fullk', 'ledgerk'), **kw):
        stock = make_job(self.root, run_ids[0], plans[0], 'stock150t', 'stock', **kw)
        full = make_job(self.root, run_ids[1], plans[1], 'full150tra24', 'full', release=True, **kw)
        ledger = make_job(self.root, run_ids[2], plans[2], 'ledger150ta24', 'ledger_query', release=True,
                          ledger=True, **kw)
        return stock, full, ledger

    def test_percentile_nearest_rank(self):
        self.assertEqual(rt.percentile([5, 1, 3, 2, 4], 50), 3)
        self.assertEqual(rt.percentile(list(range(1, 11)), 90), 9)
        self.assertEqual(rt.percentile(list(range(1, 11)), 91), 10)
        self.assertIsNone(rt.percentile([], 50))
        self.assertTrue(rt.dist([], 'nothing')['p50'].startswith('n/a (nothing'))

    def test_three_jobs_side_by_side(self):
        stock, full, ledger = self.three()
        out = self.root / 'tables'
        # given out of order: the output is stock, full, ledger_query; job and run directories both accepted
        code, text = run(['--out', out, ledger[0], full[1], stock[0]])
        self.assertEqual(code, 0, text)
        self.assertIn('cross_checks_failed=0', text)
        for name in ('summary.md', 'summary.json', 'per-job.csv', 'per-instance.csv'):
            self.assertIn('NOT AUDITED', (out / name).read_text(encoding='utf-8'), name)
        value = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(value['audit'], 'NOT AUDITED')
        self.assertEqual([j['metrics']['policy'] for j in value['jobs']], ['stock', 'full', 'ledger_query'])
        self.assertEqual(value['cross_checks_failed'], 0)
        s, f, q = (j['metrics'] for j in value['jobs'])
        records = instance_records('x', 'full', 6, 4)
        walls = sorted(r['wall_s'] for r in records)
        self.assertEqual(f['resolved'], 4)
        self.assertEqual(f['jct_s_p50'], walls[2])  # nearest rank: ceil(0.5 x 6) = 3rd
        self.assertEqual(f['jct_s_p90'], walls[5])
        self.assertEqual(f['jct_s_max'], walls[5])
        span = max(r['end'] for r in records) - min(r['start'] for r in records)
        self.assertAlmostEqual(f['makespan_s'], span, places=2)
        calls = sum(r['calls'] for r in records)
        self.assertAlmostEqual(f['requests_per_s'], calls / span, places=3)
        self.assertEqual(f['turns_per_instance_max'], 9)
        self.assertEqual(f['format_errors'], 3)
        self.assertEqual(f['rollbacks'], 3)
        self.assertEqual(f['capacity_waits'], 6)  # sessions 1, 3, 5: 2 refusals each
        self.assertEqual(f['capacity_wait_requests'], 3)
        self.assertEqual((f['evictions'], f['readmissions'], f['evicted_pages']), (3, 1, 21))
        self.assertEqual((f['dropped_by_rebuild'], f['pending_at_end']), (1, 1))
        self.assertEqual(f['prompt_tok_per_turn_mean'], 1000.0)
        self.assertEqual(f['gen_tok_per_call_p50'], 60)
        self.assertEqual(f['gen_tok_per_s_steady'], 30.0)
        self.assertEqual(f['requests_per_s_steady'], 0.5)
        self.assertEqual(s['stock_preemptions'], 7)  # the last sample's counter, not the check's quoted 5
        self.assertEqual((s['vllm_preemptions'], f['vllm_preemptions']), (7, 0))
        self.assertEqual(f['vllm_requests_finished'], calls)  # sample-10 is the last (numeric, not text, order)
        self.assertEqual(q['ledger']['parity'], 'PASS')
        self.assertEqual(q['ledger']['parity_requests'], q['active_selection_turns'])
        # no silent zeros: what a job cannot have is n/a with its reason
        for key in ('rollbacks', 'capacity_waits', 'evictions', 'readmissions', 'prompt_tok_per_call_p50',
                    'active_selection_turns', 'ledger'):
            self.assertTrue(str(s[key]).startswith('n/a ('), (key, s[key]))
        self.assertTrue(f['ledger'].startswith('n/a (policy full'))
        self.assertTrue(q['stock_preemptions'].startswith('n/a ('))
        per = {j['tag']: j['per_instance'] for j in value['jobs']}
        full_tag = value['order'][1]
        self.assertEqual(per[full_tag]['repo__repo-000']['evicted'], 2)
        self.assertEqual(per[full_tag]['repo__repo-000']['readmitted'], 1)
        self.assertEqual(per[full_tag]['repo__repo-001']['evicted'], 1)
        self.assertEqual(per[full_tag]['repo__repo-002']['evicted'], 0)
        self.assertEqual(per[full_tag]['repo__repo-001']['rollbacks'], 1)
        self.assertEqual(per[full_tag]['repo__repo-001']['capacity_waits'], 2)
        with (out / 'per-instance.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(len(rows), 7)  # header + 6 instances
        self.assertEqual(rows[0][0], 'instance_id')
        self.assertTrue(rows[0][1].startswith(value['order'][0] + ':'))
        self.assertTrue(all(r[-1] == 'NOT AUDITED' for r in rows[1:]))
        with (out / 'per-job.csv').open(newline='', encoding='utf-8') as stream:
            jobs = list(csv.DictReader(stream))
        self.assertEqual([j['policy'] for j in jobs], ['stock', 'full', 'ledger_query'])
        self.assertTrue(all(j['audit'] == 'NOT AUDITED' for j in jobs))
        md = (out / 'summary.md').read_text(encoding='utf-8')
        self.assertIn('Side by side (NOT AUDITED)', md)
        self.assertNotIn('WARNING', md)
        # the core archive was streamed, nothing extracted next to it
        self.assertEqual(sorted(p.name for p in (self.root / 'evidence' / 'swe32-9000002-r1').iterdir()),
                         ['swe32-core.tar.xz'])

    def test_without_core_evidence(self):
        stock, full, ledger = self.three()
        out = self.root / 'tables'
        code, text = run(['--out', out, '--no-evidence', stock[0], full[0]])
        self.assertEqual(code, 0, text)
        value = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
        s, f = value['jobs']
        self.assertTrue(s['metrics']['stock_preemptions'].startswith('n/a (core samples not read; the check '
                                                                     'quotes at least 5'))
        self.assertTrue(f['metrics']['vllm_preemptions'].startswith('n/a ('))
        self.assertEqual(f['metrics']['evictions'], 3)  # the join's totals stay
        self.assertTrue(f['per_instance']['repo__repo-000']['evicted'].startswith('n/a (core archive not read'))

    def test_explicit_core_and_old_runtime(self):
        """--core RUN_ID=PATH; a runtime without rollback rows says so instead of 0."""
        run_dir, job_dir = make_job(self.root, 'swe32-9000004-r1', 'fullk', 'full150tra24', 'full', release=True,
                                    thinking=False)
        moved = self.root / 'elsewhere.tar.xz'
        shutil.move(str(self.root / 'evidence' / 'swe32-9000004-r1' / 'swe32-core.tar.xz'), str(moved))
        out = self.root / 'tables'
        code, text = run(['--out', out, '--core', 'swe32-9000004-r1=%s' % moved, job_dir])
        self.assertEqual(code, 0, text)
        f = json.loads((out / 'summary.json').read_text(encoding='utf-8'))['jobs'][0]
        self.assertEqual(f['per_instance']['repo__repo-000']['evicted'], 2)
        self.assertTrue(f['metrics']['rollbacks'].startswith('n/a (runtime before keep-rebuild-1116'))
        self.assertTrue(f['per_instance']['repo__repo-001']['rollbacks'].startswith('n/a ('))

    def test_cross_check_failure_is_flagged(self):
        stock, full, ledger = self.three()
        path = full[1] / ('join-done-' + STAMP) / 'join.json'
        join = json.loads(path.read_text(encoding='utf-8'))
        join['servers'][0]['calls'] += 1
        path.write_text(json.dumps(join), encoding='utf-8')
        out = self.root / 'tables'
        code, text = run(['--out', out, full[0]])
        self.assertEqual(code, 0, text)
        self.assertIn('cross_checks_failed=1', text)
        self.assertIn('WARNING: 1 cross-check(s) failed', (out / 'summary.md').read_text(encoding='utf-8'))

    def test_run150_refused_while_absent(self):
        out = self.root / 'tables'
        code, text = run(['--out', out, '--run150', '--verify-root', self.root])
        self.assertEqual(code, 2)
        self.assertIn('RUN150_TABLES_REFUSED RUN150 not ready', text)
        for _, _, run_id, plan in rt.RUN150:
            self.assertIn(run_id, text)
        self.assertFalse(out.exists())
        # two of three present: still refused, naming the absent one only
        make_job(self.root, rt.RUN150[0][2], rt.RUN150[0][3], 'stock150t', 'stock')
        make_job(self.root, rt.RUN150[1][2], rt.RUN150[1][3], 'full150tra24', 'full', release=True)
        code, text = run(['--out', out, '--run150', '--verify-root', self.root])
        self.assertEqual(code, 2)
        self.assertIn(rt.RUN150[2][2], text)
        self.assertNotIn(rt.RUN150[0][2] + ' (', text)
        self.assertFalse(out.exists())

    def test_run150_side_by_side(self):
        self.three(run_ids=tuple(r[2] for r in rt.RUN150), plans=tuple(r[3] for r in rt.RUN150))
        out = self.root / 'tables'
        code, text = run(['--out', out, '--run150', '--verify-root', self.root])
        self.assertEqual(code, 0, text)
        value = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(value['order'], ['stock', 'full', 'ledger_query'])
        self.assertEqual(value['mode'], 'run150')
        self.assertIn('| Metric | stock | full | ledger_query |', (out / 'summary.md').read_text(encoding='utf-8'))

    def test_run150_refuses_a_wrong_plan(self):
        self.three(run_ids=tuple(r[2] for r in rt.RUN150), plans=('stockkeep150k24', 'other', 'ledger150k24'))
        code, text = run(['--out', self.root / 'tables', '--run150', '--verify-root', self.root])
        self.assertEqual(code, 2)
        self.assertIn('expected plan full150k24', text)

    def test_refusals(self):
        stock, _, _ = self.three()
        out = self.root / 'tables'
        out.mkdir()
        code, text = run(['--out', out, stock[0]])
        self.assertEqual((code, 'exists' in text), (2, True))
        code, text = run(['--out', self.root / 'new', self.root / 'nothing-here'])
        self.assertEqual(code, 2)
        code, text = run(['--out', self.root / 'new2'])
        self.assertIn('give job directories or --run150', text)
        code, text = run(['--out', self.root / 'new3', '--core', 'x=/no/such.tar.xz', stock[0]])
        self.assertEqual(code, 2)


if __name__ == '__main__':
    unittest.main()
