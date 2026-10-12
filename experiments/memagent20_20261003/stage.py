"""Start one unchanged policy server and run a resumable benchmark segment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import urlopen

from tebench.config import ROOT, load
from tebench.serve import command, environment


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def snapshot_metrics(config, stage, label):
    """Save native engine counters outside the timed benchmark interval."""
    endpoint = config['backend']['endpoint'].rstrip('/')
    if endpoint.endswith('/v1'):
        endpoint = endpoint[:-3]
    try:
        with urlopen(endpoint+'/metrics', timeout=10) as response:
            payload = response.read().decode('utf-8')
        (stage/(label+'.prom')).write_text(payload)
        write(stage/(label+'.json'), dict(time=time.time(), captured=True))
    except Exception as exc:
        # Monitoring must never turn a successful inference into a failure.
        write(stage/(label+'.json'), dict(time=time.time(), captured=False,
                                        error=str(exc)))


def verify(bundle):
    frozen = json.loads((bundle/'algorithm_sources.json').read_text())
    for name, digest in frozen.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    protocol = bundle/'protocol_sources.json'
    if protocol.exists():
        for name, digest in json.loads(protocol.read_text()).items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name


def engine_failed(config):
    """Detect a dead EngineCore even while its HTTP process drains requests."""
    endpoint = config['backend']['endpoint'].rstrip('/')
    if endpoint.endswith('/v1'):
        endpoint = endpoint[:-3]
    try:
        with urlopen(endpoint+'/health', timeout=5):
            return False
    except HTTPError as exc:
        return exc.code == 503  # Native vLLM reports EngineDeadError this way.
    except OSError:
        return False  # A transient timeout does not establish engine failure.


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--budget',required=True)
    p.add_argument('--segment',required=True)
    p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--limit',type=int)
    p.add_argument('--case-id')
    p.add_argument('--validation',action='store_true')
    a = p.parse_args()
    verify(a.bundle)
    c = load(a.bundle/(a.budget+'.yaml'))
    output = a.bundle/('validation' if a.validation else 'runs')/(a.method+'-'+a.budget)
    stage = output/('stage-'+a.segment)
    stage.mkdir(parents=True,exist_ok=False)
    env = environment(a.method,c)
    env.update(HF_HUB_OFFLINE='1',VLLM_SERVER_DEV_MODE='0',
               VLLM_AGENTKV_MODEL_AUDIT='0',VLLM_REPAIRKV_VERIFY='0',VLLM_REPAIRKV_AUDIT='0',
               VLLM_AGENTKV_VERIFY='0',VLLM_RANDOM_ATTENTION_VERIFY='0',VLLM_COMPACT_WINDOW_VERIFY='0')
    for key in ('VLLM_INTENTKV_STATS','VLLM_AGENTKV_STATS','VLLM_RANDOM_ATTENTION_STATS','VLLM_COMPACT_WINDOW_STATS'):
        env[key] = str(stage/'policy-events.jsonl')
    cmd = command(c,a.method)
    write(stage/'launch.json',dict(configuration=c,command=cmd,method=a.method,budget=a.budget,
        started=time.time(),job=os.environ.get('SLURM_JOB_ID'),deadline=a.deadline,
        algorithm_sources_sha256=hashlib.sha256((a.bundle/'algorithm_sources.json').read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((a.bundle/'manifest.json').read_bytes()).hexdigest()))
    with (stage/'server.log').open('w') as log:
        server = subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline = time.time()+1200
            while time.time()<deadline:
                if server.poll() is not None:
                    raise RuntimeError(f'Inference server exited with {server.returncode}; see {stage}/server.log')
                try:
                    with urlopen(c['backend']['endpoint']+'/models',timeout=5) as response:
                        assert response.status == 200
                        break
                except Exception:
                    time.sleep(2)
            else:
                raise TimeoutError('Server did not become ready within 1200 seconds')
            from scripts.run_clean_stage import native_smoke
            native_smoke(c,a.method,stage/'transport-smoke.json')
            snapshot_metrics(c,stage,'engine-before-benchmark')
            runner = [sys.executable,'-B','-u',str(ROOT/'experiments/memagent20_20261003/run.py'),
                      '--bundle',str(a.bundle),'--method',a.method,'--budget',a.budget,
                      '--output',str(output),'--deadline',str(a.deadline)]
            if a.limit:
                runner += ['--limit',str(a.limit)]
            if a.case_id:
                runner += ['--case-id',a.case_id]
            run = subprocess.Popen(runner,env=env)
            next_health_check = time.monotonic() + 30
            while run.poll() is None:
                failed = server.poll() is not None
                if time.monotonic() >= next_health_check:
                    failed = failed or engine_failed(c)
                    next_health_check = time.monotonic() + 30
                if failed:
                    run.terminate()
                    try:
                        run.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        run.kill()
                        run.wait()
                    raise RuntimeError('Inference engine failed during the benchmark')
                time.sleep(5)
            if run.returncode:
                raise RuntimeError(f'Benchmark runner exited with {run.returncode}')
            snapshot_metrics(c,stage,'engine-after-benchmark')
            verify(a.bundle)
            metrics = json.loads((output/'metrics.json').read_text())
            write(stage/'completed.json',dict(passed=True,time=time.time(),metrics=metrics))
        finally:
            if server.poll() is None:
                os.killpg(server.pid,signal.SIGTERM)
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid,signal.SIGKILL)
                    server.wait()


if __name__ == '__main__':
    main()
