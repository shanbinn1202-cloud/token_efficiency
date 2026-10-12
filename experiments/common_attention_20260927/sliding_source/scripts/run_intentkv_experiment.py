"""Run validated IntentKV smoke and/or aligned 20-case benchmark stages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen
from tebench.config import ROOT, load
from tebench.methods import check_support
from tebench.serve import command, environment


def verify_sources(bundle):
    manifest = json.loads((bundle/'implementation_manifest.json').read_text())
    for relative, digest in manifest['sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Source changed after validation preparation: {relative}')
    original = json.loads((bundle/'upstream_before.json').read_text())
    for relative, digest in original['sha256'].items():
        assert hashlib.sha256((ROOT/'baselines/IntentKV'/relative).read_bytes()).hexdigest() == digest


def stage(bundle, label):
    verify_sources(bundle)
    smoke = label == 'smoke'
    prerequisite = bundle/('gpu_kernels.json' if smoke else 'model_smoke.json')
    assert json.loads(prerequisite.read_text())['passed'], prerequisite
    config_path = bundle/('16k.yaml' if smoke else f'{label}.yaml')
    config = load(config_path)
    check_support('intentkv', config)
    reference = json.loads((bundle/'vanilla_reference.json').read_text())
    for section in ('model','generation','dataset','evaluation','backend'):
        assert config[section] == reference['configuration'][section], section
    env = environment('intentkv', config)
    env['HF_HUB_OFFLINE'] = '1'
    stats = bundle/f'{label}-events.jsonl'
    if stats.exists():
        raise RuntimeError(f'Existing stage events: {stats}; preserve this attempt and use a new bundle')
    env['VLLM_INTENTKV_STATS'] = str(stats)
    env['VLLM_SERVER_DEV_MODE'] = '1' if smoke else '0'
    stamp = os.environ.get('SLURM_JOB_ID',str(os.getpid()))
    started = time.time()
    with (bundle/f'server-{label}-{stamp}.log').open('w') as log:
        server = subprocess.Popen(command(config,'intentkv'),env=env,stdout=log,
                                  stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline = time.monotonic()+1200
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Server {label} exited: {server.returncode}')
                try:
                    with urlopen(config['backend']['endpoint']+'/models',timeout=2):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('vLLM startup timeout')
                    time.sleep(2)
            if smoke:
                cmd = [sys.executable,'-B','-u','scripts/validate_intentkv_model.py',
                    '--config',str(config_path),'--stats',str(stats),'--output',str(bundle/'model_smoke.json')]
            else:
                cmd = [sys.executable,'-B','-u','-m','tebench.run','--method','intentkv',
                    '--config',str(config_path),'--output',str(bundle/label)]
            subprocess.run(cmd,check=True)
            verify_sources(bundle)
            if not smoke:
                run = json.loads((bundle/label/'run.json').read_text())
                assert run['instance_ids'] == reference['instance_ids']
                metrics = json.loads((bundle/label/'metrics.json').read_text())
                assert metrics['eligible_for_comparison']
            (bundle/f'{label}-completed.json').write_text(json.dumps(
                dict(stage=label,job_id=stamp,started=started,finished=time.time(),passed=True),indent=2))
        finally:
            try:
                os.killpg(server.pid,signal.SIGTERM)
                server.wait(timeout=30)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(server.pid,signal.SIGKILL)
                server.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True,type=Path)
    p.add_argument('--stage',choices=('all','smoke','16k','12k','8k'),default='all')
    args = p.parse_args()
    for label in (('smoke','16k','12k','8k') if args.stage=='all' else (args.stage,)):
        print(f'INTENTKV_STAGE_START {label}',flush=True)
        stage(args.bundle,label)
        print(f'INTENTKV_STAGE_COMPLETED {label}',flush=True)


if __name__ == '__main__':
    main()
