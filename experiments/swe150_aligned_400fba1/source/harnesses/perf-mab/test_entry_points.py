"""The delivered payload as the stage runs it (local only): every sealed file from dist/payload.json written into a
fresh directory, imported with `python -I -B` (no harness directory on sys.path, no bytecode), the controller's,
the job's and the runner's entry points present, the runner's own process refusing a bad spec, job.sh's
interpreter line and unsets, bootstrap's nonce check, and no private data in the payload (the real MemAgent32 data
is pinned by hashes only)."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.ensure()

PAYLOAD = json.loads((ROOT / 'dist' / 'payload.json').read_bytes())


class EntryPoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.stage = Path(cls.tmp.name) / 'stage'
        cls.stage.mkdir()
        for name, item in PAYLOAD['files'].items():
            raw = base64.b64decode(item['base64'])
            assert hashlib.sha256(raw).hexdigest() == item['sha256']
            (cls.stage / name).write_bytes(raw)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_isolated(self, code, *args):
        return subprocess.run([sys.executable, '-I', '-B', '-c', code, str(self.stage), *args], capture_output=True,
                              text=True, timeout=120, cwd=str(self.stage.parent))

    def test_payload_files_equal_the_sources(self):
        for name, item in PAYLOAD['files'].items():
            self.assertEqual(item['sha256'], hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), name)
        self.assertEqual(sorted(PAYLOAD['files']), sorted([
            'bootstrap.py', 'controller.py', 'job.py', 'job.sh', 'mab_checks.py', 'mab_metrics.py', 'mab_runner.py',
            'mab_workload.py', 'measure.py', 'pins.json', 'public-release.bundle', 'qualification_ports.py',
            'qualification_shutdown.py', 'resources.py', 'stock.py', 'transport.ps1']))

    def test_isolated_imports(self):
        code = ('import sys; sys.path.insert(0, sys.argv[1]); import job, controller, mab_runner, mab_workload, '
                'mab_checks, mab_metrics, stock, measure, resources, qualification_ports, qualification_shutdown; '
                'import json; pins = json.load(open(sys.argv[1] + "/pins.json")); '
                'assert mab_workload.build_plan(pins["config"], pins.get("real_data")) == pins["workload_plan"]; '
                'assert callable(job.main) and callable(controller.setup) and callable(controller.qualify); '
                'assert callable(mab_runner.main); '
                'print("IMPORTS_OK", sorted(m for m in sys.modules if m in ("job", "mab_runner")))')
        result = self.run_isolated(code)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn('IMPORTS_OK', result.stdout)
        self.assertFalse(list(self.stage.rglob('__pycache__')))

    def test_runner_process(self):
        bad = self.stage.parent / 'bad-spec.json'
        bad.write_text(json.dumps(dict(schema='other')), encoding='utf-8')
        result = subprocess.run([sys.executable, '-I', '-B', str(self.stage / 'mab_runner.py'), str(bad)],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 2)
        self.assertIn('unknown runner spec schema', result.stderr)
        self.assertFalse(list(self.stage.rglob('__pycache__')))

    def test_job_sh(self):
        text = (self.stage / 'job.sh').read_text(encoding='utf-8')
        self.assertIn('exec "$clean/bin/python" -I -B "$stage/job.py" "$stage"', text)
        self.assertIn('unset VLLM_KV_CACHE_LAYOUT VLLM_CUSTOM_SCOPES_FOR_PROFILING REMOTE_RA_STEP_PROFILE '
                      'REMOTE_RA_STEP_PROFILE_EVERY', text)
        self.assertIn('export VLLM_USE_FLASHINFER_SAMPLER=0', text)
        self.assertIn('export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1', text)
        self.assertNotIn('\r', text)

    def test_bootstrap_refuses_a_bad_nonce(self):
        code = ('import sys; sys.path.insert(0, sys.argv[1]); import bootstrap\n'
                'try:\n    bootstrap.receive(sys.argv[1], "XYZ", b"{}")\nexcept ValueError as e:\n    print("REFUSED", e)')
        result = self.run_isolated(code)
        self.assertIn('REFUSED invalid nonce', result.stdout, result.stderr)
        self.assertEqual(json.loads((self.stage / 'pins.json').read_bytes())['slurm']['job_name_prefix'], 'pmab-')
        self.assertIn("STAGE_PREFIX = 'pmab-'", (self.stage / 'bootstrap.py').read_text(encoding='utf-8'))

    def test_no_private_data(self):
        pins = json.loads((self.stage / 'pins.json').read_bytes())
        real = pins.get('real_data') or {}
        self.assertEqual(set(real), {'manifest_sha256', 'count', 'cases', 'deltaai_dir', 'source'})
        for case in real.get('cases') or ():
            self.assertEqual(set(case), {'id', 'context_sha256', 'chunks', 'context_tokens', 'split'})
        self.assertNotIn('query', json.dumps(pins['workload_plan']))


if __name__ == '__main__':
    unittest.main()
