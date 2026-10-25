"""Reuse the established full-history protocol with first-allocation telemetry.

No data is selected here. The fixed balanced manifest is prepared after SWE.
Official text scoring is deferred to grade_memagent, outside the timed window.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time

import httpx

from tebench.backends import create
from tebench.config import load
from experiments.memagent20_20261003.run import run_case
from experiments.benchmarks32_20261004.telemetry import LOCAL, instrument_backend, write


class DeferredScorer:
    @staticmethod
    def post_process(*args, **kwargs):
        return {}, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--budget', required=True)
    a = p.parse_args()
    data = a.bundle/'memagent'
    manifest = json.loads((data/'manifest.json').read_text())
    cases = manifest['cases']
    assert len(cases) == manifest['count']
    config = load(data/(a.budget+'.yaml'))
    assert len(cases) == config['dataset']['count']
    import tebench.config as config_module
    assert Path(config_module.__file__).resolve().is_relative_to(Path('/projects/bcjw/ssun4/token_efficiency/experiments/gemma_full_20261017/harness'))
    write(a.stage/'runner-imports.json', dict(config_module=config_module.__file__, expected=len(cases)))
    a.output.mkdir(parents=True, exist_ok=True)
    prediction_path = a.output/'predictions.json'
    predictions = {r['id']: r for r in json.loads(prediction_path.read_text())} if prediction_path.exists() else {}
    pending = [c for c in cases if c['id'] not in predictions]
    backend = instrument_backend(create(config, a.method), a.stage/'requests.jsonl')
    root_url = config['backend']['endpoint'].removesuffix('/v1')
    try:
        backend.complete(model=config['model']['name'], messages=[{'role':'user', 'content':'Reply OK.'}], max_tokens=2)
        backend.reset()
        with httpx.Client(timeout=10) as control:
            response = control.post(root_url+'/benchmark-begin')
            response.raise_for_status()
            write(a.stage/'measurement-start.json', response.json())

        def run(case):
            LOCAL.case = case['id']
            try:
                result = run_case(case, data, a.stage/'results', config, a.method,
                                  backend, DeferredScorer())
                result.update(grading_pending=True, stage=a.stage.name)
                return result
            finally:
                LOCAL.case = ''

        with ThreadPoolExecutor(max_workers=config['evaluation']['workers']) as pool:
            for future in as_completed([pool.submit(run, c) for c in pending]):
                result = future.result()
                iid = result['id']
                if result['status'] == 'error':
                    write(a.stage/(iid+'.error.json'), result)
                else:
                    source = a.stage/'results/cases'/iid
                    target = a.output/'cases'/iid
                    target.mkdir(parents=True, exist_ok=True)
                    for path in source.iterdir():
                        if path.name != 'result.json':
                            shutil.copy2(path, target/path.name)
                    write(target/'result.json', result)
                    predictions[iid] = dict(id=iid, status=result['status'], stage=a.stage.name)
                    write(prediction_path, [predictions[c['id']] for c in cases if c['id'] in predictions])
                print(json.dumps(dict(event='CASE_COMMITTED', id=iid, status=result['status'],
                                      completed=len(predictions), expected=len(cases))), flush=True)
        with httpx.Client(timeout=10) as control:
            response = control.post(root_url+'/benchmark-freeze')
            response.raise_for_status()
            write(a.stage/'measurement-final.json', response.json())
        write(a.stage/'runner-completed.json', dict(time=time.time(), completed=len(predictions), expected=len(cases)))
        if list(a.stage.glob('*.error.json')):
            raise RuntimeError('Infrastructure/request errors require inspection before continuation')
    finally:
        backend.close()


if __name__ == '__main__':
    main()
