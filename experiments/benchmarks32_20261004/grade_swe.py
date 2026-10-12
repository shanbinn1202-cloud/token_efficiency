"""Cache each official SWE grade independently of GPU inference allocations."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from baselines.common.harness import run_apptainer_harness
from tebench.config import load
from experiments.benchmarks32_20261004.telemetry import write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    config = load(a.bundle/'16k.yaml')
    ids = json.loads((a.bundle/'swebench-manifest.json').read_text())['instance_ids']
    predictions = json.loads((a.output/'predictions.json').read_text())
    pending = [row for row in predictions if not (a.output/'grades'/row['instance_id']/'report.json').exists()]
    def grade(row):
        directory = a.output/'grades'/row['instance_id']
        directory.mkdir(parents=True,exist_ok=True)
        run_apptainer_harness(predictions=[row],dataset_name=config['dataset']['name'],split='test',
            instance_ids=[row['instance_id']],max_workers=1,timeout=config['evaluation']['timeout'],
            run_id=a.output.name+'-'+row['instance_id'],report_path=directory/'report.pending.json',
            executable=config['evaluation']['apptainer'],image_manifest=config['evaluation']['image_manifest'])
        # Publish the completion marker atomically. A Slurm cutoff while the
        # harness writes JSON must leave this case eligible for continuation.
        report = json.loads((directory/'report.pending.json').read_text())
        write(directory/'report.json', report)
        return row['instance_id']
    with ThreadPoolExecutor(max_workers=4) as pool:
        for done in as_completed([pool.submit(grade,row) for row in pending]):
            print('GRADED',done.result(),flush=True)
    reports = [json.loads(p.read_text()) for p in (a.output/'grades').glob('*/report.json')]
    resolved = sorted({iid for report in reports for iid in report.get('resolved_ids',[])})
    errors = sorted({iid for report in reports for iid in report.get('infrastructure_error_ids',[])})
    write(a.output/'accuracy.json',dict(expected=32,graded=len(reports),resolved=len(resolved),
        accuracy=len(resolved)/32 if len(reports)==32 and not errors else None,
        resolved_ids=resolved,infrastructure_errors=errors,instance_ids=ids))
    if errors:
        raise RuntimeError('Grading infrastructure failed: '+','.join(errors))


if __name__ == '__main__':
    main()
