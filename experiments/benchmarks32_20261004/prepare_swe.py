"""Freeze the first 32 reference-valid tasks without reusing predictions."""
import hashlib
import json
from pathlib import Path
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path('/projects/bcjw/ssun4/benchmark32_restart_20261004')
SOURCE = Path('/projects/bcjw/ssun4/benchmark40_final_20261004')
METHODS = ['vanilla', 'random_attention', 'sliding_window', 'agentdiet', 'agentkv', 'intentkv']


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def main():
    assert not BUNDLE.exists(), 'Never overwrite a frozen experiment bundle'
    algorithms = json.loads((SOURCE/'algorithm_sources.json').read_text())
    for name, digest in algorithms.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    original = json.loads((SOURCE/'swebench-manifest.json').read_text())
    gold = json.loads((SOURCE/'gold-validation/accuracy.json').read_text())
    assert gold['graded']==40 and gold['resolved']==40 and not gold['infrastructure_errors']
    rows = original['instances'][:32]
    ids = [r['instance_id'] for r in rows]
    assert len(set(ids))==32 and set(ids).issubset(gold['resolved_ids'])
    BUNDLE.mkdir(parents=True)
    manifest = dict(original, count=32, instances=rows, instance_ids=ids,
        sampling='First 32 in the previously frozen reference-valid seed-42 order; no model outcomes used.',
        source_manifest=str(SOURCE/'swebench-manifest.json'),
        source_manifest_sha256=hashlib.sha256((SOURCE/'swebench-manifest.json').read_bytes()).hexdigest(),
        reused_model_predictions=False)
    write(BUNDLE/'swebench-manifest.json', manifest)
    images = json.loads((SOURCE/'images/manifest.json').read_text())
    images.update(instance_ids=ids, images={iid:images['images'][iid] for iid in ids})
    assert all(Path(entry['image']).is_file() for entry in images['images'].values())
    (BUNDLE/'images').mkdir()
    write(BUNDLE/'images/manifest.json', images)
    for budget in ('16k','12k','8k'):
        config = yaml.safe_load((SOURCE/(budget+'.yaml')).read_text())
        config['dataset']['count'] = 32
        config['backend']['endpoint'] = 'http://127.0.0.1:42632/v1'
        config['evaluation']['image_manifest'] = str(BUNDLE/'images/manifest.json')
        (BUNDLE/(budget+'.yaml')).write_text(yaml.safe_dump(config, sort_keys=False))
    calibration = BUNDLE/'gold-validation'
    (calibration/'grades').mkdir(parents=True)
    for iid in ids:
        (calibration/'grades'/iid).symlink_to(SOURCE/'gold-validation/grades'/iid, target_is_directory=True)
    write(calibration/'accuracy.json', dict(expected=32, graded=32, resolved=32,
        accuracy=1.0, resolved_ids=ids, instance_ids=ids, infrastructure_errors=[],
        reused_reference_checks=32, fresh_reference_checks=0,
        note='Reuse verified reference-patch checks with identical images and environment; no inference predictions reused.'))
    write(BUNDLE/'reference-check.json', dict(ready=True, reference_total=32,
        reference_fully_resolved=32, known_reference_failures={},
        environment_sha256=algorithms['baselines/common/environment.py'],
        reference_source=str(calibration/'accuracy.json')))
    write(BUNDLE/'algorithm_sources.json', algorithms)
    version = json.loads((SOURCE/'source-version.json').read_text())
    version.update(recorded=time.time(), count_per_benchmark=32,
        code='Latest six-method algorithms unchanged; asynchronous harness with 32 tasks and one run each for Vanilla/AgentDiet.')
    write(BUNDLE/'source-version.json', version)
    write(BUNDLE/'restart.json', dict(time=time.time(), reason='User requested all methods restart fresh with 32 tasks on each benchmark',
        cancelled_job='3305938', previous_bundle=str(SOURCE), reused_predictions=False,
        reused_images=True, reused_reference_checks=32))
    write(BUNDLE/'plan.json', dict(methods=METHODS, budgets=['16k','12k','8k'],
        benchmark_order=['swebench','memagent'], count_per_benchmark=32,
        memagent_count_per_ability=8, memagent_selection='after SWE phase',
        workers=16, context_tokens=262144, kv_pool_bytes=20*1024**3,
        allocation_seconds=7200, fresh_allocation_per_group=True,
        swe_groups=14, memagent_groups=14, budget_independent_methods=['vanilla','agentdiet'],
        throughput='Main-agent output tokens / measured first-allocation wall time; includes unfinished requests, excludes startup, warmup and grading; auxiliary tokens separate.',
        accuracy='Finish all 32 fixed cases, continuing on fresh allocations if required.'))
    for name in ('vllm-cache','inductor-cache','triton-cache'):
        (BUNDLE/name).symlink_to(SOURCE/name, target_is_directory=True)
    summary = (SOURCE/'summarize.py').read_text().replace('== 40','== 32').replace('expected=40','expected=32')
    (BUNDLE/'summarize.py').write_text(summary)
    files = list(Path(__file__).parent.glob('*.py'))
    files += [ROOT/'experiments/memagent20_20261003'/name for name in ('run.py','judge.py','allocation.py')]
    files += [ROOT/'experiments/memagent40_20261003/prepare.py']
    files += list((ROOT/'external/MemoryAgentBench').rglob('*.py'))
    files += [BUNDLE/name for name in ('swebench-manifest.json','reference-check.json','images/manifest.json','16k.yaml','12k.yaml','8k.yaml')]
    sources = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write(BUNDLE/'protocol_sources.json', sources)
    write(BUNDLE/'prepared-swe.json', dict(ready=True, time=time.time(), count=32, ids=ids))
    print(json.dumps(dict(bundle=str(BUNDLE), count=32, reference_passed=32, groups_per_benchmark=14), indent=2))


if __name__=='__main__':
    main()
