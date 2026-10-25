"""Exact-source control trials for the merged candidate, never formal results."""
import asyncio
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

R = Path(__file__).resolve().parent
D = R.parents[1]
sys.path.insert(0, str(D))
from qualify_terminal_images import prepare, Trial, TrialConfig, TaskConfig, AgentConfig, EnvironmentConfig
sys.path.insert(0, str(R))


def digest():
    names = json.loads((R / 'source-manifest.json').read_text())
    actual = {n: hashlib.sha256((R / n).read_bytes()).hexdigest() for n in names}
    if actual != names:
        raise RuntimeError('Candidate runtime changed after sealing')
    return hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()


async def one(name, agent):
    output = R / 'controls' / (name + '-' + agent + '.json')
    if output.exists():
        return
    before = digest()
    image = D / 'terminal-images' / (name + '.sif')
    image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
    cfg = TrialConfig(task=TaskConfig(path=D/'terminal-arm64-tasks'/name),
        trial_name=name+'-native-kvm-apt-v42-'+agent+'-'+str(time.time_ns()),
        trials_dir=D/'terminal-validation', agent=AgentConfig(name=agent),
        environment=EnvironmentConfig(import_path='terminal_environment:Arm64ApptainerEnvironment'))
    try:
        trial = await Trial.create(cfg)
        result = await trial.run()
        row = json.loads(result.model_dump_json())
        reward = result.verifier_result.rewards if result.verifier_result else None
        row['passed'] = result.exception_info is None and reward is not None and reward.get('reward') == (1 if agent == 'oracle' else 0)
    except Exception:
        row = dict(passed=False, traceback=traceback.format_exc())
    row.update(task=name, agent=agent, runtime_sha256=before, image_sha256=image_hash,
               finished=time.time(), formal=False)
    if before != digest() or image_hash != hashlib.sha256(image.read_bytes()).hexdigest():
        row.update(passed=False, source_changed=True)
    output.write_text(json.dumps(row, indent=2))
    print(json.dumps({k:row[k] for k in ['task','agent','passed','finished']}), flush=True)


async def main():
    import fcntl
    lock = (R / 'controls.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (R / 'controls').mkdir(exist_ok=True)
    digest()
    pairs = set(json.loads((D/'revisions/terminal-candidate-status.json').read_text())['candidate_paired_passes'])
    pairs.update(['adaptive-rejection-sampler','db-wal-recovery','count-dataset-tokens','install-windows-3.11'])
    names = sorted(p.name for p in (D/'terminal-bench-2').iterdir() if (p/'task.toml').exists())
    first = ['build-pmars','build-pov-ray','adaptive-rejection-sampler','mteb-retrieve','caffe-cifar-10','cobol-modernization','install-windows-3.11','db-wal-recovery','count-dataset-tokens','qemu-startup','headless-terminal','constraints-scheduling','make-doom-for-mips','rstan-to-pystan']
    names = first + [n for n in names if n not in first]
    sem = asyncio.Semaphore(4)
    async def task(name):
        async with sem:
            await one(name, 'oracle')
            if name in pairs:
                await one(name, 'nop')
    await asyncio.gather(*(task(n) for n in names))
    print('Merged candidate controls finished; review all failures before promotion.', flush=True)


if __name__ == '__main__':
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed)
    asyncio.run(main())
