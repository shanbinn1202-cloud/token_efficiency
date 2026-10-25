"""Official per-instance grading, using the frozen full dataset and isolated workers."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path

from tebench.config import load
from experiments.benchmarks32_20261004.telemetry import write


def grade(arguments):
    bundle, output, row = arguments
    from baselines.common import harness
    from arm64_log_compat import install
    install()
    config = load(bundle / '8k.yaml')
    manifest = json.loads((bundle / 'swebench-manifest.json').read_text())
    def dataset(name, split):
        assert name == manifest['dataset'] and split == manifest['split']
        return manifest['instances']
    # Only the data-loading boundary changes; official test scripts and scoring
    # stay intact. Processes avoid the earlier concurrent HF/tqdm loader race.
    harness.load_dataset = dataset
    directory = output / 'grades' / row['instance_id']
    directory.mkdir(parents=True, exist_ok=True)
    harness.run_apptainer_harness(predictions=[row], dataset_name=config['dataset']['name'],
        split='test', instance_ids=[row['instance_id']], max_workers=1,
        timeout=config['evaluation']['timeout'], run_id=output.name+'-'+row['instance_id'],
        report_path=directory/'report.pending.json', executable=config['evaluation']['apptainer'],
        image_manifest=config['evaluation']['image_manifest'])
    report = json.loads((directory/'report.pending.json').read_text())
    report['evaluated_patch_sha256'] = hashlib.sha256(row.get('model_patch', '').encode()).hexdigest()
    report['log_parser_compatibility'] = 'arm64_log_compat.py: non-strict ARM XPASS recognition for xarray-4356; official labels unchanged'
    write(directory/'report.json', report)
    return row['instance_id']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = load(args.bundle/'8k.yaml')
    ids = json.loads((args.bundle/'swebench-manifest.json').read_text())['instance_ids']
    predictions = json.loads((args.output/'predictions.json').read_text())
    assert len({r['instance_id'] for r in predictions}) == len(predictions)
    assert {r['instance_id'] for r in predictions} <= set(ids)
    pending = []
    for row in predictions:
        report_path = args.output/'grades'/row['instance_id']/'report.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            assert report['evaluated_patch_sha256'] == hashlib.sha256(row.get('model_patch', '').encode()).hexdigest()
        else:
            pending.append((args.bundle, args.output, row))
    with ProcessPoolExecutor(max_workers=config['evaluation']['grading_workers'],
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(grade, item) for item in pending]):
            print('GRADED', future.result(), flush=True)
    reports = [json.loads((args.output/'grades'/r['instance_id']/'report.json').read_text()) for r in predictions]
    resolved = sorted({iid for report in reports for iid in report.get('resolved_ids', [])})
    errors = sorted({iid for report in reports for iid in report.get('infrastructure_error_ids', [])})
    write(args.output/'accuracy.json', dict(expected=len(ids), graded=len(reports), resolved=len(resolved),
        accuracy=len(resolved)/len(ids) if len(reports)==len(ids) and not errors else None,
        resolved_ids=resolved, infrastructure_errors=errors, instance_ids=ids))
    if errors:
        raise RuntimeError('Grading infrastructure failed: '+','.join(errors))


if __name__ == '__main__':
    main()
