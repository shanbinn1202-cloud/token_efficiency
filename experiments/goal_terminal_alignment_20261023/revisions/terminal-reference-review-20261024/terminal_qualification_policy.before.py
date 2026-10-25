"""Keep reference failures distinct from the user's approved evaluation exception."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
APPROVAL = R / 'revisions/terminal-protein-reference-exception-20261024/approval.json'


def require_coverage(proof):
    if not proof.get('ready_for_formal') or proof.get('expected') != 89 or proof.get('checked') != 89:
        raise RuntimeError('All 89 native environments must be reviewed before formal evaluation')
    names = {p.parent.name for p in (R / 'terminal-bench-2').glob('*/task.toml')}
    passed = set(proof.get('reference_passed_tasks', []))
    exceptions = set(proof.get('approved_reference_exceptions', []))
    if len(names) != 89 or passed & exceptions or passed | exceptions != names:
        raise RuntimeError('Reference coverage does not match the original 89 tasks')
    if exceptions:
        if exceptions != {'protein-assembly'}:
            raise RuntimeError('Unapproved reference exception')
        approval = json.loads(APPROVAL.read_text())
        if approval.get('authorized_by') != 'user' or approval.get('task') != 'protein-assembly':
            raise RuntimeError('Missing explicit user approval')
        task = R / 'terminal-bench-2/protein-assembly'
        actual = {str(p.relative_to(task)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in task.rglob('*') if p.is_file()}
        if actual != approval['original_task_files_sha256']:
            raise RuntimeError('Approved original task changed')
    if not proof.get('negative_controls_passed'):
        raise RuntimeError('Negative controls incomplete')

