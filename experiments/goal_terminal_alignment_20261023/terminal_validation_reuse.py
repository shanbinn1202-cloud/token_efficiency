"""Reuse a completed GPU flow only across an explicitly reviewed non-model delta.

Incomplete later smoke trials remain incomplete. This cannot approve a failed
reference solution; the separate all-task coverage gate must still succeed.
"""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
REVIEW = R / 'revisions/terminal-standard-dev-v46/integration/completed-gpu-delta-review.json'


def reviewed_evidence(passed, sources):
    if not REVIEW.exists():
        return None
    review = json.loads(REVIEW.read_text())
    if str(passed) != review['completed_gpu_trial']:
        return None
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(passed) != review['completed_gpu_trial_sha256']:
        return None
    start = passed.parent / 'source-at-start.json'
    before = json.loads(start.read_text())
    changed = {n: {'before': before.get(n), 'after': sources.get(n)}
               for n in before.keys() | sources.keys() if before.get(n) != sources.get(n)}
    allowed = {str(R / n) for n in ['terminal-runtime/kvm_environment.py',
                                   'terminal_stage.py', 'terminal_qualification_policy.py']}
    if not changed or not set(changed) <= allowed or changed != review['changed_files']:
        return None
    evidence = [REVIEW, Path(__file__), start]
    for name, expected in review['regression_evidence_sha256'].items():
        p = Path(name)
        if sha(p) != expected or not json.loads(p.read_text()).get('passed'):
            return None
        evidence.append(p)
    if len(review['regression_evidence_sha256']) != 6:
        return None
    return evidence
