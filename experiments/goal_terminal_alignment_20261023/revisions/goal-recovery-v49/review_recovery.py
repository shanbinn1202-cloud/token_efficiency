"""Review a narrow lifecycle delta against retained native GPU evidence.

This does not claim a fresh GPU run. New allocations must pass the real
16-session above-budget probe before opening their formal timing window.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REV = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from fingerprints import goal

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    for name in ['grading-preflight.json', 'judge-preflight.json', 'package-validation.json',
                 'known-baseline-test-issue.json', 'original-source-archive-reviewed.json']:
        assert json.loads((REV/name).read_text())['passed'], name
    unit = (REV/'unit-tests.log').read_text()
    assert 'Ran 11 tests' in unit and unit.rstrip().endswith('OK')
    compatibility = (REV/'compatibility-reviewed.log').read_text()
    assert 'Ran 20 tests' in compatibility
    assert 'FAILED (failures=1, skipped=2)' in compatibility
    assert compatibility.count('FAIL: test_a_rebuild_with_every_window_slot_taken_releases_idle_pages') == 1
    assert '(345, 345, 25)' in compatibility
    assert '\nERROR:' not in compatibility
    changes = json.loads((REV/'source-revision.json').read_text())['changes']
    allowed = {
        ROOT/'selective-prefix/source-overlay'/item['file'] for item in changes
    } | {ROOT/name for name in ['grade_goal.py', 'goal_stage.py', 'stage_goal_runtime.py',
                               'preflight_imports.py', 'selective-prefix/INSTALL_RELEASE.json',
                               'selective-prefix/SOURCE_SHA256SUMS.txt']}
    expected_payload = {'remote_ra_linux/native_authority.py', 'registered_add_protocol.py',
                        'serving_controller.py', 'remote_ra_live_v2/idle_release.py',
                        'registered_add_bridge.py'}
    assert {item['file'] for item in changes} == expected_payload
    current = goal()
    evidence = [REV/name for name in ['source-revision.json', 'unit-tests.log',
        'compatibility-reviewed.log', 'compatibility-baseline-window.log',
        'known-baseline-test-issue.json', 'grading-preflight.json', 'judge-preflight.json',
        'package-validation.json', 'original-source-archive-reviewed.json',
        'grading-dependencies-manifest.json', 'review_recovery.py', 'run_compatibility.py']]
    evidence += [ROOT/'test_goal_recovery_v49.py']
    records = []
    for model in ['qwen', 'gemma']:
        previous = json.loads((REV/'before/qualification'/f'goal-{model}.json').read_text())
        assert previous['passed']
        assert set(previous['source_sha256']).issubset(current), 'Frozen source coverage lost'
        changed = {Path(name) for name, pin in previous['source_sha256'].items()
                   if current[name] != pin}
        assert changed == allowed, (model, changed ^ allowed)
        # All selector, Page scoring, tensor, attention, GDN and model/config
        # files are still exactly those covered by the retained GPU proof.
        for name, pin in previous['evidence_sha256'].items():
            assert digest(name) == pin, 'Original GPU evidence changed: '+name
        extra = set(current)-set(previous['source_sha256'])
        assert extra == {str(ROOT/'goal_recovery_probe.py'), str(ROOT/'fingerprints.py'),
                         str(REV/'grading-dependencies-manifest.json')}, extra
        data = dict(previous)
        data.update(time=time.time(), source_sha256=current,
                    lifecycle_recovery_revision='goal-recovery-v49',
                    base_gpu_validation_reused_after_review=True,
                    fresh_gpu_validation_completed=False,
                    mandatory_gpu_16session_probe_before_formal_measurement=True,
                    review_limit='11 fault tests;17 compatibility passes;2 existing optional skips;'
                                 '1 verified pre-existing fake-window boundary assertion. '
                                 'Native CUDA kernels and all algorithm/config pins unchanged.')
        data['evidence_sha256'] = dict(previous['evidence_sha256'])
        data['evidence_sha256'].update({str(p):digest(p) for p in evidence})
        target = ROOT/'qualification'/f'goal-{model}.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2));temporary.replace(target)
        records.append(dict(model=model, changed_sources=len(changed), source_files=len(current)))
    assert 'probe(server,stage)' in (ROOT/'goal_stage.py').read_text()
    assert (ROOT/'goal_stage.py').read_text().index('probe(server,stage)') < (
        ROOT/'goal_stage.py').read_text().index('run_goal_harness.py')
    (REV/'review-passed.json').write_text(json.dumps(dict(
        passed=True, time=time.time(), models=records,
        fresh_gpu_validation_completed=False,
        mandatory_fresh_gpu_check_before_formal_measurement=True,
        retained_original_gpu_evidence=True), indent=2))
    print('Reviewed lifecycle recovery; fresh native probe remains mandatory.', flush=True)

if __name__ == '__main__':
    main()
