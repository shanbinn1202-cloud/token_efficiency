"""Request a fresh two-hour allocation on a different node from the last one."""
import json

from experiments.memagent20_20261003.allocation import ALLOC


def allocation_command(bundle):
    records = [json.loads(path.read_text())
               for path in bundle.glob('hardware-*.json')]
    latest = max(records, key=lambda row: row['allocation_start'], default=None)
    command = list(ALLOC)
    if latest and latest.get('node'):
        command.insert(command.index('srun'), '--exclude=' + latest['node'])
    return command
