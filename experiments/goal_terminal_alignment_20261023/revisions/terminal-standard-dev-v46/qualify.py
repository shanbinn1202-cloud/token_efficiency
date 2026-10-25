"""Review complete controls before releasing the unchanged 89-task benchmark."""
import hashlib
import json
import shutil
import sys
import time
import tomllib
from pathlib import Path

R = Path(__file__).resolve().parent
D = R.parents[1]
sys.path.insert(0, str(D))
from terminal_qualification_policy import APPROVAL, require_coverage, approval_path


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def checked_image_digest(p):
    cache = R / 'image-integrity.json'
    if cache.exists():
        data = json.loads(cache.read_text())
        row = data.get('images', {}).get(p.stem, {})
        st = p.stat()
        stamp = {'resolved': str(p.resolve()), 'device': st.st_dev, 'inode': st.st_ino,
                 'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'ctime_ns': st.st_ctime_ns}
        if data.get('complete') and row.get('stamp') == stamp:
            return row['sha256']
    return digest(p)


def main():
    expected = json.loads((R / 'expected-controls.json').read_text())
    manifest = json.loads((R / 'source-manifest.json').read_text())
    assert all(digest(R / n) == h for n, h in manifest.items()), 'Candidate changed'
    assert all(digest(D / 'terminal-runtime' / n) == h for n, h in manifest.items()), 'Integrated runtime changed'
    runtime_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    baseline = D / 'revisions/terminal-native-kvm-apt-v42'
    old_manifest = json.loads((baseline / 'source-manifest.json').read_text())
    assert all(digest(baseline / n) == h for n, h in old_manifest.items()), 'Prior tested runtime changed'
    assert [n for n in manifest if manifest[n] != old_manifest[n]] == ['kvm_environment.py']
    added = ('/busybox ln -sfn /proc/self/fd /dev/fd\n'
             '/busybox ln -sfn /proc/self/fd/0 /dev/stdin\n'
             '/busybox ln -sfn /proc/self/fd/1 /dev/stdout\n'
             '/busybox ln -sfn /proc/self/fd/2 /dev/stderr\n')
    old_code = (baseline / 'kvm_environment.py').read_text()
    new_code = (R / 'kvm_environment.py').read_text()
    assert new_code.count(added) == 1 and new_code.replace(added, '') == old_code
    old_runtime_hash = hashlib.sha256(json.dumps(old_manifest, sort_keys=True).encode()).hexdigest()
    evidence_paths = {}
    rows = {}
    pending = []
    failures = []
    exceptions = []
    for agent, names in expected.items():
        for name in names:
            p = R / 'controls' / (name + '-' + agent + '.json')
            expected_runtime_hash = runtime_hash
            if not p.exists():
                p = baseline / 'controls' / p.name
                expected_runtime_hash = old_runtime_hash
            if not p.exists():
                pending.append(p.name)
                continue
            evidence_paths[name, agent] = p
            row = json.loads(p.read_text())
            if row.get('pending'):
                pending.append(p.name)
                continue
            if row.get('runtime_sha256') != expected_runtime_hash or row.get('source_changed'):
                raise RuntimeError('Control source mismatch: ' + p.name)
            rows[name, agent] = row
            if agent == 'nop':
                trial = D / 'terminal-validation' / row['trial_name']
                ctrf = trial / 'verifier/ctrf.json'
                if not ctrf.exists():
                    failures.append(p.name + ': missing executed-test report')
                    continue
                results = json.loads(ctrf.read_text()).get('results', {})
                summary = results.get('summary', {})
                if summary.get('tests', 0) < 1 or summary.get('failed', 0) < 1 or not results.get('tests'):
                    failures.append(p.name + ': negative control did not execute failing tests')
                    continue
            if row.get('passed'):
                continue
            if name == 'protein-assembly' and agent == 'oracle' and row.get('exception_info') is None:
                trial = D / 'terminal-validation' / row['trial_name']
                oracle = (trial / 'agent/oracle.txt').read_text(errors='replace')
                verifier = (trial / 'verifier/test-stdout.txt').read_text(errors='replace')
                reward = (row.get('verifier_result') or {}).get('rewards', {}).get('reward')
                if reward == 0 and 'EnforceTranslation' in oracle and 'Wrong translation' in oracle and 'test_gblock' in verifier:
                    exceptions.append(name)
                    continue
            if name in {'crack-7z-hash', 'portfolio-optimization'} and agent == 'oracle':
                try:
                    approval_path(name)
                except (RuntimeError, FileNotFoundError):
                    failures.append(p.name + ': awaiting explicit user decision')
                    continue
                trial = D / 'terminal-validation' / row['trial_name']
                if name == 'crack-7z-hash':
                    error = row.get('exception_info') or {}
                    valid_exception = (error.get('exception_type') == 'AgentTimeoutError'
                                       and '900.0 seconds' in error.get('exception_message', ''))
                else:
                    ctrf = json.loads((trial / 'verifier/ctrf.json').read_text())['results']
                    failed_tests = [t for t in ctrf['tests'] if t['status'] == 'failed']
                    raw_report = (trial / 'verifier/test-stdout.txt').read_text(errors='replace')
                    raw_failures = [line for line in raw_report.splitlines() if line.startswith('FAILED ')]
                    valid_exception = (row.get('exception_info') is None
                        and ctrf['summary']['passed'] == 3 and ctrf['summary']['failed'] == 1
                        and len(failed_tests) == 1
                        and failed_tests[0]['name'].endswith('::test_performance_and_scalability')
                        and len(raw_failures) == 3
                        and all(any('::test_performance_and_scalability[' + str(n) + ']' in line
                                    for line in raw_failures) for n in (5000, 6000, 8000)))
                if valid_exception:
                    exceptions.append(name)
                    continue
            failures.append(p.name)
    if pending or failures:
        print(json.dumps({'ready': False, 'completed_controls': len(rows), 'pending': pending, 'failures': failures}))
        return
    for name in ['log-summary-date-ranges', 'qemu-alpine-ssh', 'db-wal-recovery']:
        for agent in ['oracle', 'nop']:
            assert evidence_paths[name, agent].parent == R / 'controls' and rows[name, agent]['passed']
    reference_passed = [n for n in expected['oracle'] if rows[n, 'oracle']['passed']]
    proof = {
        'time': time.time(), 'ready_for_formal': True,
        'all_tasks_validated': not exceptions,
        'expected': 89, 'checked': len(expected['oracle']),
        'passed': len(reference_passed), 'reference_passed_tasks': reference_passed,
        'approved_reference_exceptions': exceptions,
        'negative_controls_passed': all(rows[n, 'nop']['passed'] for n in expected['nop']),
        'negative_controls_count': len(expected['nop']),
        'runtime_files_sha256': manifest, 'runtime_sha256': runtime_hash,
        'validation_strategy': 'Full prior runtime controls plus exact four-standard-symlink delta review and targeted Oracle/NOP regressions on log-summary-date-ranges, qemu-alpine-ssh and db-wal-recovery; separate fresh GPU validation binds integrated final runtime.',
        'control_runtime_sources': {n+'-'+a: str(p) for (n,a),p in evidence_paths.items()},
        'note': 'All original tasks retained. Approved reference failure is not a passing reference or a model result.',
    }
    require_coverage(proof)
    source_proofs = {}
    for name in expected['oracle']:
        image = D / 'terminal-images' / (name + '.sif')
        if checked_image_digest(image) != rows[name, 'oracle']['image_sha256']:
            raise RuntimeError('Image changed: ' + name)
        src = D / 'terminal-bench-2' / name
        dst = D / 'terminal-arm64-tasks' / name
        original = tomllib.loads((src / 'task.toml').read_text())
        original['environment']['docker_image'] = str(image)
        if original != tomllib.loads((dst / 'task.toml').read_text()):
            raise RuntimeError('Task configuration changed: ' + name)
        hashes = {str(p.relative_to(src)): digest(p) for p in src.rglob('*') if p.is_file() and p.name != 'task.toml'}
        if any(digest(dst / n) != h for n, h in hashes.items()):
            raise RuntimeError('Original task files changed: ' + name)
        source_proofs[name] = {'files': hashes, 'task_config_changes': ['environment.docker_image only'],
                               'original_task_toml_sha256': digest(src / 'task.toml')}
    proof['control_evidence_sha256'] = {str(p): digest(p) for p in evidence_paths.values()}
    if exceptions:
        proof['approval_sha256'] = {str(approval_path(n)): digest(approval_path(n)) for n in exceptions}
    out = D / 'terminal-images'
    current = json.loads((out / 'qualification.json').read_text())
    if current.get('ready_for_formal'):
        if current.get('runtime_sha256') != runtime_hash or current.get('control_evidence_sha256') != proof['control_evidence_sha256']:
            raise RuntimeError('Existing released qualification differs; explicit revision required')
        print('Already qualified; immutable released evidence preserved')
        return
    backup = R / 'integration/previous-qualification'
    if not backup.exists():
        shutil.copytree(out / 'qualification', backup)
        shutil.copy2(out / 'qualification.json', backup / 'aggregate-before.json')
    for (name, agent), row in rows.items():
        (out / 'qualification' / (name + '-' + agent + '.json')).write_text(json.dumps(row, indent=2))
    for name, source in source_proofs.items():
        (out / 'qualification' / (name + '-source.json')).write_text(json.dumps(source, indent=2))
    tmp = out / 'qualification.reviewed.tmp'
    tmp.write_text(json.dumps(proof, indent=2))
    tmp.replace(out / 'qualification.json')
    print(json.dumps({'ready': True, 'reference_passed': len(reference_passed), 'approved_exceptions': exceptions,
                      'negative_controls': len(expected['nop']), 'formal_task_count': 89}))


if __name__ == '__main__':
    main()
