"""MemAgent stage, matching SWE timing, telemetry and allocation boundaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx

from tebench.config import ROOT, load
from tebench.serve import command, environment
from experiments.benchmarks32_20261004.telemetry import write


def verify(bundle):
    for name in ('algorithm_sources.json', 'protocol_sources.json'):
        for path, digest in json.loads((bundle/name).read_text()).items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path


def stop(process, grace=10):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def count_predictions(output):
    path = output/'predictions.json'
    return len(json.loads(path.read_text())) if path.exists() else 0


def inference(a, config, output, stage, env):
    cmd = command(config, a.method)
    observer = 'experiments.benchmarks32_20261004.telemetry.middleware'
    # vLLM consumes one import path per --middleware occurrence. Register the
    # observer last so it also wraps routes handled by the policy middleware.
    cmd += ['--middleware', observer]
    if '--attention-backend' not in cmd:
        cmd += ['--attention-backend', 'FLASH_ATTN']
    if '--attention-config' not in cmd:
        cmd += ['--attention-config', json.dumps({'flash_attn_version':3})]
    env['BENCHMARK_LEDGER_PATH'] = str(stage/'ledger.json')
    for key in ('VLLM_INTENTKV_STATS','VLLM_AGENTKV_STATS',
                'VLLM_RANDOM_ATTENTION_STATS','VLLM_COMPACT_WINDOW_STATS'):
        env[key] = str(stage/'policy-events.jsonl')
    write(stage/'launch.json', dict(command=cmd, configuration=config,
        method=a.method, budget=a.budget, job=os.environ['SLURM_JOB_ID'],
        deadline=a.deadline, time=time.time(), source_sha256=hashlib.sha256(
            (a.bundle/'algorithm_sources.json').read_bytes()).hexdigest()))
    root = config['backend']['endpoint'].removesuffix('/v1')
    runner = None
    cutoff = False
    with (stage/'server.log').open('w') as log, httpx.Client(timeout=10) as client:
        server = subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,
                                  start_new_session=True)
        try:
            startup_deadline = min(time.time()+1200, a.deadline-180)
            while time.time()<startup_deadline:
                if server.poll() is not None:
                    raise RuntimeError('Server failed during startup')
                try:
                    response = client.get(root+'/health')
                    response.raise_for_status()
                    break
                except httpx.HTTPError:
                    time.sleep(2)
            else:
                raise TimeoutError('Server startup timed out')
            from experiments.benchmarks32_20261004.validate_transport import validate
            validate(config, a.method, stage)
            (stage/'engine-before.prom').write_text(client.get(root+'/metrics').text)
            args = [sys.executable,'-B','-u','-m','experiments.benchmarks32_20261004.run_memagent',
                    '--bundle',str(a.bundle),'--output',str(output),'--stage',str(stage),
                    '--method',a.method,'--budget',a.budget]
            with (stage/'runner-stdout.log').open('w') as runner_log:
                runner = subprocess.Popen(args,env=env,stdout=runner_log,
                    stderr=subprocess.STDOUT,start_new_session=True)
                next_metrics = 0
                while runner.poll() is None:
                    if time.time() >= a.deadline-35:
                        cutoff = True
                        break
                    if server.poll() is not None:
                        raise RuntimeError('Server process failed during inference')
                    if time.time() >= next_metrics:
                        health = client.get(root+'/health')
                        health.raise_for_status()
                        metrics = client.get(root+'/metrics')
                        metrics.raise_for_status()
                        with (stage/'engine-metrics.jsonl').open('a') as handle:
                            handle.write(json.dumps(dict(time=time.time(),metrics=metrics.text))+'\n')
                        next_metrics = time.time()+15
                    time.sleep(1)
                measured = client.post(root+'/benchmark-freeze')
                measured.raise_for_status()
                measured = measured.json()
                write(stage/'measurement-final.json', measured)
                assert measured['started'] is not None, 'Runner never started measurement'
                assert not measured['tokens'].get('untagged', 0), measured['tokens']
                assert not measured['tokens'].get('warmup', 0), measured['tokens']
                first = output/'first-allocation-throughput.json'
                if a.attempt == 1:
                    assert not first.exists(), 'First allocation result is immutable'
                    write(first, dict(measured, job=os.environ['SLURM_JOB_ID'],
                        cutoff=cutoff, stage=str(stage), completed=count_predictions(output)))
                stop(runner)
                (stage/'engine-after.prom').write_text(client.get(root+'/metrics').text)
                if not cutoff and runner.returncode:
                    raise RuntimeError(f'Runner failed: {runner.returncode}')
                if not cutoff:
                    expected = {}
                    for line in (stage/'requests.jsonl').read_text().splitlines():
                        row = json.loads(line)
                        if row['kind'] != 'warmup' and 'usage' in row:
                            expected[row['kind']] = expected.get(row['kind'],0)+row['usage']['completion_tokens']
                    assert measured['tokens'] == expected, (measured['tokens'],expected)
        finally:
            stop(runner)
            stop(server, grace=15)
    return cutoff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--method',required=True)
    parser.add_argument('--budget',required=True)
    parser.add_argument('--attempt',type=int,required=True)
    parser.add_argument('--deadline',type=float,required=True)
    a = parser.parse_args()
    verify(a.bundle)
    config = load(a.bundle/'memagent'/(a.budget+'.yaml'))
    output = a.bundle/'memagent'/(a.method+'-'+a.budget)
    stage = output/f'allocation-{a.attempt:02d}'
    stage.mkdir(parents=True,exist_ok=False)
    env = environment(a.method, config)
    env.update(HF_HUB_OFFLINE='1',VLLM_SERVER_DEV_MODE='0',
        VLLM_AGENTKV_MODEL_AUDIT='0',VLLM_AGENTKV_VERIFY='0',
        VLLM_RANDOM_ATTENTION_VERIFY='0',VLLM_COMPACT_WINDOW_VERIFY='0')
    # Own all temporary sandboxes; no accumulated state leaks into later tasks.
    scratch = Path('/tmp')/f"benchmark32-restart-{os.environ['SLURM_JOB_ID']}"
    scratch.mkdir(parents=True,exist_ok=True)
    env.update(TMPDIR=str(scratch),APPTAINER_TMPDIR=str(scratch),
               APPTAINER_CACHEDIR=str(scratch/'image-cache'),
               VLLM_CACHE_ROOT=str(a.bundle/'vllm-cache'),
               TORCHINDUCTOR_CACHE_DIR=str(a.bundle/'inductor-cache'),
               TRITON_CACHE_DIR=str(a.bundle/'triton-cache'))
    cutoff = False
    try:
        if count_predictions(output)<32:
            cutoff = inference(a,config,output,stage,env)
        if count_predictions(output)==32 and time.time()<a.deadline-120:
            with (stage/'grading.log').open('w') as log:
                grader = subprocess.Popen([sys.executable,'-B','-u','-m',
                    'experiments.benchmarks32_20261004.grade_memagent',
                    '--bundle',str(a.bundle),'--output',str(output)],env=env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    while grader.poll() is None and time.time()<a.deadline-35:
                        time.sleep(2)
                    if grader.poll() is not None and grader.returncode:
                        raise RuntimeError(f'Grading failed: {grader.returncode}')
                finally:
                    stop(grader)
        verify(a.bundle)
        write(stage/'completed.json',dict(time=time.time(),cutoff=cutoff,
              predictions=count_predictions(output),accuracy=(json.loads(
              (output/'accuracy.json').read_text()) if (output/'accuracy.json').exists() else None)))
    except Exception as exc:
        write(stage/'failed.json',dict(time=time.time(),error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
