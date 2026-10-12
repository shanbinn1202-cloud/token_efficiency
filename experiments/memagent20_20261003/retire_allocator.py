"""Retire the suspended allocator only after its retained GPU job has ended."""
import json
from pathlib import Path
import subprocess
import time

AUDIT = Path('/u/ssun4/token_efficiency_results/memagent20_20261003/audit/repairkv-excluded-by-user')
while True:
    result = subprocess.run(['squeue', '-h', '-j', '3304549', '-o', '%i'],
                            capture_output=True, text=True)
    if not result.stdout.strip():
        # Confirm disappearance independently before retiring the PTY owner.
        checked = subprocess.run(['scontrol', 'show', 'job', '3304549'],
                                 capture_output=True, text=True)
        if 'Invalid job id specified' in checked.stderr or any(
                'JobState='+s in checked.stdout
                for s in ['COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT']):
            break
    time.sleep(30)
subprocess.run(['systemctl', '--user', 'stop',
                'memagent20-supervisor-20261003.service'], check=True)
(AUDIT/'allocator-retired.json').write_text(json.dumps(dict(time=time.time(),
    job='3304549', unit='memagent20-supervisor-20261003.service'), indent=2))
