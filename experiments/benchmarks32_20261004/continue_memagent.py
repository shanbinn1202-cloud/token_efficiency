"""Wait for SWE completion, then prepare the balanced subset and supervise it."""
import json
import subprocess
import sys
import time

from experiments.benchmarks32_20261004.prepare_swe import BUNDLE
from experiments.benchmarks32_20261004.telemetry import write


def main():
    while True:
        path = BUNDLE/'supervisor-state.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        if state.get('state')=='swebench_complete_memagent_selection_pending':
            break
        time.sleep(30)
    write(BUNDLE/'pipeline-state.json', dict(state='preparing_memagent32', time=time.time()))
    if not (BUNDLE/'memagent/prepared.json').exists():
        subprocess.run([sys.executable, '-B', '-u', '-m',
                        'experiments.benchmarks32_20261004.prepare_memagent'], check=True)
    from experiments.benchmarks32_20261004.prepare_memagent import assert_swe_complete
    assert_swe_complete()
    from experiments.benchmarks32_20261004.stage import verify
    verify(BUNDLE)
    write(BUNDLE/'pipeline-state.json', dict(state='running_memagent32', time=time.time()))
    subprocess.run([sys.executable, '-B', '-u', '-m',
                    'experiments.benchmarks32_20261004.supervise_memagent'], check=True)
    write(BUNDLE/'pipeline-state.json', dict(state='inference_complete_check_summary_judge', time=time.time()))


if __name__=='__main__':
    main()
