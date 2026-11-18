"""The verifier and the tables on a synthetic g1 stage (fakes), and the forgeries they must catch: a changed file
(evidence-sha256), a consistently re-hashed transcript (the recomputed token check), a changed server status, a
changed outcome, the wrong Slurm state, a synthetic archive verified as a real one, a bulk archive that differs."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402
import fake_server as fs  # noqa: E402
import mab_tables  # noqa: E402
import synthetic_stage  # noqa: E402
import verify_evidence as ve  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Postrun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        saved = {k: getattr(fs.FakeProcess, k) for k in fs.FakeProcess.KNOBS}
        try:
            for key, value in synthetic_stage.G1_KNOBS.items():
                setattr(fs.FakeProcess, key, value)
            pins = fs.synthetic_pins(c.load(ROOT / 'pins.json'))
            cls.stage = cls.root / ('pmab-' + 'c' * 32)
            cls.status, cls.remote = fs.build_stage(cls.stage, '3999998', pins, plan='g1')
        finally:
            for key, value in saved.items():
                setattr(fs.FakeProcess, key, value)
        cls.archive = cls.stage / 'fetch' / 'pmab-core.tar.xz'
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def out(self):
        type(self).n += 1
        return self.root / ('check-%d' % self.n)

    def verify(self, archive=None, **kwargs):
        kwargs.setdefault('synthetic', True)
        kwargs.setdefault('slurm_state', 'COMPLETED')
        kwargs.setdefault('slurm_exit_code', '0:0')
        return ve.verify(archive or self.archive, '3999998', self.remote, self.out(), **kwargs)

    def repack(self, change, rehash):
        """The core archive with `change(files)` applied; with `rehash`, evidence-sha256.json updated to match."""
        work = self.root / ('repack-%d' % self.n)
        self.n += 1
        names = ve.extract(self.archive, work)
        change(work)
        if rehash:
            listing = json.loads((work / 'evidence-sha256.json').read_bytes())
            listing['core'] = {n: sha(work / n) for n in listing['core']}
            (work / 'evidence-sha256.json').write_text(json.dumps(listing, sort_keys=True, indent=2) + '\n')
        target = work.parent / (work.name + '.tar.xz')
        with tarfile.open(target, 'w:xz') as stream:
            for name in names:
                stream.add(work / name, arcname=name, recursive=False)
        return target

    def test_clean_archive_and_tables(self):
        self.assertEqual(self.status, 'PMAB_PASS')
        out = self.out()
        value = ve.verify(self.archive, '3999998', self.remote, out, synthetic=True, slurm_state='COMPLETED',
                          slurm_exit_code='0:0', bulk_dir=self.stage / 'fetch')
        self.assertEqual(value['verdict'], 'PMAB_PASS_VERIFIED', value['failed'])
        self.assertIn('bulk', [r['check'] for r in value['checks']])
        written = mab_tables.write_tables(out)
        self.assertIn('tables.md', [p.name for p in written])
        servers = (out / 'mab-tables' / 'servers.csv').read_text(encoding='utf-8')
        self.assertIn('g1a', servers)
        self.assertEqual(self.verify(slurm_state=None, slurm_exit_code=None)['verdict'],
                         'ARCHIVE_VERIFIED_SLURM_EXIT_UNCONFIRMED')

    def test_forgeries(self):
        def edit_requests(files):
            path = files / 'job-output' / 'g1a' / 'client' / 'requests.jsonl'
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            rows[1]['response']['output_ids'][0] += 1
            path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows), encoding='utf-8')
        value = self.verify(self.repack(edit_requests, rehash=False))
        self.assertEqual(value['verdict'], 'NOT_VERIFIED')
        self.assertIn('evidence-sha256', value['failed'])
        value = self.verify(self.repack(edit_requests, rehash=True))
        self.assertEqual(value['verdict'], 'NOT_VERIFIED')
        self.assertTrue(any(f.startswith('checks-g1a') or f.startswith('check-details-g1a') for f in value['failed']),
                        value['failed'])

        def edit_status(files):
            path = files / 'summary-g1b.json'
            summary = json.loads(path.read_bytes())
            summary['status'] = 'PASS_TRUNCATED'
            path.write_text(json.dumps(summary, sort_keys=True, indent=2) + '\n')
        value = self.verify(self.repack(edit_status, rehash=True))
        self.assertIn('status-g1b', value['failed'])

        def edit_result(files):
            path = files / 'result.json'
            result = json.loads(path.read_bytes())
            result['status'] = 'PMAB_PARTIAL_PASS'
            path.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
        value = self.verify(self.repack(edit_result, rehash=True))
        self.assertIn('outcome', value['failed'])
        self.assertEqual(self.verify(slurm_state='FAILED', slurm_exit_code='3:0')['verdict'], 'NOT_VERIFIED')
        self.assertIn('pins', self.verify(synthetic=False)['failed'])

    def test_bulk_forgery(self):
        bulk = self.root / 'bulk-forged'
        bulk.mkdir()
        for item in (self.stage / 'fetch').glob('pmab-bulk-*.tar.xz'):
            shutil.copyfile(item, bulk / item.name)
        work = self.root / 'bulk-work'
        names = ve.extract(bulk / 'pmab-bulk-g1a.tar.xz', work)
        runtime = work / 'job-output' / 'g1a' / 'runtime.jsonl'
        rows = [json.loads(line) for line in runtime.read_text(encoding='utf-8').splitlines()]
        rows[0]['prompt_token_ids'][0] += 1
        runtime.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        with tarfile.open(bulk / 'pmab-bulk-g1a.tar.xz', 'w:xz') as stream:
            for name in names:
                stream.add(work / name, arcname=name, recursive=False)
        value = self.verify(bulk_dir=bulk)
        self.assertIn('bulk', value['failed'])


if __name__ == '__main__':
    unittest.main()
