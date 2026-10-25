"""Keep explicitly approved reference failures distinct from passing controls."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
APPROVAL = R / 'revisions/terminal-protein-reference-exception-20261024/approval.json'
APPROVALS = {
    'protein-assembly': APPROVAL,
    'crack-7z-hash': R / 'revisions/terminal-reference-review-20261024/crack-7z-hash.approval.json',
    'portfolio-optimization': R / 'revisions/terminal-reference-review-20261024/portfolio-optimization.approval.json',
}


def approval_path(name):
    if name not in APPROVALS:
        raise RuntimeError('Unreviewed reference exception: ' + name)
    path = APPROVALS[name]
    approval = json.loads(path.read_text())
    if approval.get('authorized_by') != 'user' or approval.get('task') != name:
        raise RuntimeError('Missing explicit user approval: ' + name)
    task = R / 'terminal-bench-2' / name
    actual = {str(p.relative_to(task)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in task.rglob('*') if p.is_file()}
    if actual != approval['original_task_files_sha256']:
        raise RuntimeError('Approved original task changed: ' + name)
    return path


def require_coverage(proof):
    if not proof.get('ready_for_formal') or proof.get('expected') != 89 or proof.get('checked') != 89:
        raise RuntimeError('All 89 native environments must be reviewed before formal evaluation')
    names = {p.parent.name for p in (R / 'terminal-bench-2').glob('*/task.toml')}
    passed = set(proof.get('reference_passed_tasks', []))
    exceptions = set(proof.get('approved_reference_exceptions', []))
    if len(names) != 89 or passed & exceptions or passed | exceptions != names:
        raise RuntimeError('Reference coverage does not match the original 89 tasks')
    for name in exceptions:
        approval_path(name)
    if not proof.get('negative_controls_passed'):
        raise RuntimeError('Negative controls incomplete')
