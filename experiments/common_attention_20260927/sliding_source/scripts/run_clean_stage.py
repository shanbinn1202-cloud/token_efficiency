"""Validate and benchmark the independent clean-upstream six-method engine."""
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
from tebench.serve import command, environment


def verify_upstream(bundle):
    random_manifest = bundle/"upstream_Random-Attention.json"
    if random_manifest.exists():
        original = json.loads(random_manifest.read_text())
        for relative, expected in original["sha256"].items():
            assert hashlib.sha256((ROOT/"baselines/Random-Attention"/relative).read_bytes()).hexdigest() == expected, relative
        copied = ROOT/"adapters/random_attention/selector_random.py"
        assert copied.read_bytes() == (ROOT/"baselines/Random-Attention/scripts/vllm_rp_bench/selector_random.py").read_bytes()
    for name in ('IntentKV', 'RepairKV', 'agentkv'):
        manifest = json.loads((bundle/f'upstream_{name}.json').read_text())
        for relative, expected in manifest['sha256'].items():
            assert hashlib.sha256((ROOT/'baselines'/name/relative).read_bytes()).hexdigest() == expected
    manifest = json.loads((bundle/'upstream_agentkv_engine.json').read_text())
    for relative, expected in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/'baselines/agentkv/engine'/relative).read_bytes()).hexdigest() == expected


def native_smoke(c, method, output):
    from concurrent.futures import ThreadPoolExecutor
    from tebench.backends import create
    backend = create(c, method)
    def request(i):
        messages = [{'role':'user','content':f'Reply with the number {i}.'}]
        traces = []
        try:
            for _ in range(2):
                response = backend.complete(model=c['model']['name'], messages=messages, max_tokens=32)
                assert response.usage.completion_tokens > 0
                traces.append(dict(backend.last_measurement))
                messages += [response.choices[0].message.model_dump(exclude_none=True),
                             {'role':'user','content':'Repeat that number.'}]
            return traces
        finally:
            backend.reset()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(request, range(8)))
        output.write_text(json.dumps(dict(passed=True, concurrent_requests=8, turns=2, traces=rows),indent=2))
    finally:
        backend.close()


def verify_implementation(bundle, method=None):
    manifest = bundle/f'validated_sources-{method}.json'
    if not manifest.exists():
        manifest = bundle/'validated_sources.json'
    frozen = json.loads(manifest.read_text())
    for path, digest in frozen.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path


def early_eos_smoke(c, method, output):
    """Real EOS before max_tokens must not commit native async lookahead."""
    from concurrent.futures import ThreadPoolExecutor
    from copy import deepcopy
    from tebench.backends import create
    short = deepcopy(c)
    short['model']['enable_thinking'] = False  # Synthetic short-response prompt.
    backend = create(short, method)
    def case(i):
        messages = [{'role':'system','content':'Answer with a single number and no explanation.'},
                    {'role':'user','content':'What is 2 + 2?'}]
        turns = []
        try:
            for _ in range(2):
                response = backend.complete(model=c['model']['name'],messages=messages,max_tokens=64)
                row = dict(backend.last_measurement)
                assert response.choices[0].finish_reason == 'stop', row
                assert 0 < row['completion_tokens'] < 64 and row['native_request_finished']
                if method == 'repairkv':
                    assert row['pool_tokens'] == row['logical_computed_tokens']
                turns.append(row)
                messages += [response.choices[0].message.model_dump(exclude_none=True),
                             {'role':'user','content':'What is 3 + 3?'}]
            return turns
        finally:
            backend.reset()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            cases = list(pool.map(case,range(4)))
        output.write_text(json.dumps(dict(passed=True,cases=cases),indent=2))
    finally:
        backend.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--budget',choices=('16k','12k','8k'),default='8k')
    p.add_argument('--mode',choices=('smoke','pressure','frontend','run'),default='smoke')
    p.add_argument('--attempt',default='01')
    p.add_argument('--audit',action='store_true')
    a = p.parse_args()
    verify_upstream(a.bundle)
    cfg = a.bundle/f'{a.budget}.yaml'
    c = load(cfg)
    assert c['backend']['async_scheduling']
    reference = json.loads((a.bundle/'vanilla_reference.json').read_text())
    for section in ('model','generation','dataset','evaluation','backend'):
        expected = dict(reference['configuration'][section])
        if section == 'backend':
            expected['async_scheduling'] = True
        assert c[section] == expected, f'Configuration differs from Vanilla: {section}'
    if a.mode == 'run':
        verify_implementation(a.bundle, a.method)
    label = f'{a.method}-{a.budget}-{a.mode}-{a.attempt}'
    destination = a.bundle/label
    destination.mkdir(exist_ok=False)
    env = environment(a.method,c)
    env.update(HF_HUB_OFFLINE='1',VLLM_SERVER_DEV_MODE='1' if a.mode!='run' else '0')
    events = destination/'events.jsonl'
    env['VLLM_INTENTKV_STATS'] = env['VLLM_AGENTKV_STATS'] = str(events)
    env["VLLM_RANDOM_ATTENTION_STATS"] = str(events)
    env["VLLM_RANDOM_ATTENTION_VERIFY"] = "1" if a.mode != "run" else "0"
    env["VLLM_COMPACT_WINDOW_STATS"] = str(events)
    env["VLLM_COMPACT_WINDOW_VERIFY"] = "1" if a.mode != "run" else "0"
    env['VLLM_AGENTKV_VERIFY'] = '1' if a.mode!='run' else '0'
    env.update(VLLM_AGENTKV_MODEL_AUDIT='0',VLLM_REPAIRKV_VERIFY='0',VLLM_REPAIRKV_AUDIT='0')
    if a.audit:
        if a.method=='agentkv':
            env.update(VLLM_AGENTKV_MODEL_AUDIT='1',VLLM_AGENTKV_MODEL_AUDIT_PATH=str(destination/'algorithm-audit.json'))
        if a.method=='repairkv':
            env.update(VLLM_REPAIRKV_VERIFY='1',VLLM_REPAIRKV_AUDIT=str(destination/'audit'))
    # command() uses the audit flag to distinguish compiled model capture.
    os.environ.update({k:v for k,v in env.items() if k.startswith('VLLM_')})
    started = time.time()
    cmd = command(c,a.method)
    (destination/'launch.json').write_text(json.dumps(dict(command=cmd,configuration=c,
        method=a.method,mode=a.mode,job=os.environ.get('SLURM_JOB_ID'),started=started),indent=2))
    with (destination/'server.log').open('w') as log:
        server = subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline = time.monotonic()+1200
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited: {server.returncode}; see {destination}/server.log')
                try:
                    with urlopen(c['backend']['endpoint']+'/models',timeout=2):
                        break
                except OSError:
                    if time.monotonic()>deadline:
                        raise TimeoutError('vLLM startup timed out')
                    time.sleep(2)
            print('SERVER_READY',label,flush=True)
            if a.mode == 'pressure' and a.method in {'agentkv','repairkv'}:
                early_eos_smoke(c,a.method,destination/'early-eos.json')
            base = [sys.executable,'-B','-u']
            report = destination/'validation.json'
            if a.mode=='run':
                task = base+['-m','tebench.run','--method',a.method,'--config',str(cfg),'--output',str(destination/'results')]
            elif a.method=='random_attention':
                task = base+['scripts/validate_random_attention.py','--config',str(cfg),'--output',str(report)]
                if a.mode=='pressure':
                    task += ['--pressure']
            elif a.method=='sliding_window':
                task = base+['scripts/validate_random_attention.py','--method','sliding_window',
                             '--config',str(cfg),'--output',str(report)]
                if a.mode=='pressure':
                    task += ['--pressure']
            elif a.mode=='frontend':
                task = base+['scripts/validate_clean_frontends.py','--config',str(cfg),'--output',str(report)]
            elif a.mode=='pressure':
                task = base+['scripts/validate_clean_pressure.py','--config',str(cfg),
                             '--method',a.method,'--output',str(report)]
            elif a.method=='intentkv':
                task = base+['scripts/validate_intentkv_model.py','--config',str(cfg),'--stats',str(events),'--output',str(report)]
            elif a.method=='repairkv':
                task = base+['scripts/validate_repairkv_concurrent.py','--config',str(cfg),
                             '--output',str(report),'--sessions','8']
            elif a.method=='agentkv':
                task = base+['scripts/validate_agentkv_vllm.py','--endpoint',c['backend']['endpoint'],
                    '--budget',str(c['cache']['kv_budget']),'--interval','128','--workers','8',
                    '--rounds','3','--max-tokens','256','--initial-tokens',str(c['cache']['kv_budget']+450),
                    '--feedback-tokens','500','--output',str(report)]
            else:
                native_smoke(c,a.method,report)
                task = None
            if task:
                result = subprocess.run(task,check=False,env=env)
                if result.returncode:
                    metrics_file = destination/'results/metrics.json'
                    if a.mode!='run' or not metrics_file.exists():
                        raise subprocess.CalledProcessError(result.returncode,task)
                    measured = json.loads(metrics_file.read_text())
                    if measured.get('instances')!=20 or len(measured.get('cases',[]))!=20 or measured.get('grading_errors'):
                        raise subprocess.CalledProcessError(result.returncode,task)
                    errors=[]
                    errors_by_case={}
                    known_context=lambda e: ('ContextWindowExceededError' in e or 'Logical RoPE context limit' in e)
                    for path in (destination/'results/trajectories').glob('*.json'):
                        rows=json.loads(path.read_text()).get('memory',{}).get('requests',[])
                        case_errors=[str(row['error']) for row in rows if row.get('error')]
                        errors_by_case[path.stem]=case_errors
                        errors.extend(case_errors)
                        if any(not row.get('usage') and not known_context(str(row.get('error',''))) for row in rows):
                            raise subprocess.CalledProcessError(result.returncode,task)
                    report_data=json.loads((destination/'results/report.json').read_text())
                    invalid_case=any(case['exit_status']=='missing' or case.get('e2e_seconds') is None
                        or (case['exit_status']=='error' and not errors_by_case.get(case['instance_id']))
                        for case in measured['cases'])
                    if not errors or any(not known_context(e) for e in errors) or invalid_case or report_data.get('incomplete_instances'):
                        raise subprocess.CalledProcessError(result.returncode,task)
                    # Retain genuine model/context failures in the fixed denominator.
                    # Eligibility and every request error remain in metrics.json.
                    print('BENCHMARK_FINISHED_WITH_RECORDED_FAILURES',result.returncode,flush=True)
            verify_upstream(a.bundle)
            assert server.poll() is None, 'Inference engine exited before stage completion'
            if a.mode=='run':
                verify_implementation(a.bundle, a.method)
                actual = json.loads((destination/'results/run.json').read_text())
                reference = json.loads((a.bundle/'vanilla_reference.json').read_text())
                assert actual['instance_ids']==reference['instance_ids']
                metrics = json.loads((destination/'results/metrics.json').read_text())
                # A genuine model context-limit failure remains in the 20-task
                # denominator and does not suppress other budget experiments.
                (destination/'metrics-summary.json').write_text(json.dumps(metrics,indent=2))
            else:
                assert json.loads(report.read_text())['passed']
            (destination/'completed.json').write_text(json.dumps(dict(passed=True,started=started,
                finished=time.time(),method=a.method,budget=a.budget,mode=a.mode),indent=2))
            print('STAGE_COMPLETED',label,flush=True)
        finally:
            try:
                server.terminate()
                server.wait(timeout=30)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(server.pid,signal.SIGKILL)
                server.wait()


if __name__=='__main__':
    main()
