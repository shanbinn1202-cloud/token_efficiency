"""Run fixed independent SWE instances with the unchanged agent and policies."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import shutil
import time

import httpx

from tebench.config import load, runner_config
from tebench.backends import create
from tebench.methods import create_model
from baselines.common.environment import resolve_instance_image
from baselines.common.runner import run_instance
from experiments.benchmarks32_20261004.telemetry import LOCAL, instrument_backend, write


def budget_infeasible(row, method, budget, decisions):
    """Apply only the explicitly approved, fixed-sample infeasibility decisions."""
    for decision in decisions['cases']:
        if (row['instance_id'] == decision['instance_id']
                and method in decision['methods'] and budget == decision['budget']):
            assert decision['action'] == 'count_as_failure'
            assert decision['prefix_tokens'] >= budget - decision['reserve_tokens'][method]
            return dict(decision, method=method)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--budget',required=True)
    a = p.parse_args()
    config = load(a.bundle/(a.budget+'.yaml'))
    decisions = json.loads((a.bundle/'budget_infeasible.json').read_text())
    rows = json.loads((a.bundle/'swebench-manifest.json').read_text())['instances']
    ids = [r['instance_id'] for r in rows]
    a.output.mkdir(parents=True,exist_ok=True)
    prediction_path = a.output/'predictions.json'
    predictions = {r['instance_id']:r for r in json.loads(prediction_path.read_text())} if prediction_path.exists() else {}
    pending = [r for r in rows if r['instance_id'] not in predictions]
    for row in pending:
        row['image_name'] = resolve_instance_image(row,'',config['evaluation']['image_manifest'])
    rc = runner_config(config,a.stage/'results')
    (a.stage/'results/trajectories').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(a.stage/'runner.log')],force=True)
    os.environ.setdefault('MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT','3')
    backend = instrument_backend(create(config,a.method),a.stage/'requests.jsonl')
    root_url = config['backend']['endpoint'].removesuffix('/v1')
    try:
        backend.complete(model=config['model']['name'],messages=[{'role':'user','content':'Reply OK.'}],max_tokens=2)
        backend.reset()
        with httpx.Client(timeout=10) as control:
            response = control.post(root_url+'/benchmark-begin')
            response.raise_for_status()
            write(a.stage/'measurement-start.json',response.json())

        def run(row):
            LOCAL.case = row['instance_id']
            start = time.time()
            try:
                infeasible = budget_infeasible(row, a.method, config['cache']['kv_budget'], decisions)
                if infeasible is not None:
                    result = dict(instance_id=row['instance_id'],
                        model_name_or_path='openai/'+config['model']['name'].split('/')[-1],
                        model_patch='', exit_status='BudgetInfeasible',
                        reason='Complete fixed prefix cannot fit the configured KV budget',
                        budget_infeasibility=infeasible, started=start,
                        finished=time.time(), stage=a.stage.name)
                    # A terminal record, with no model request or token usage.
                    # The ordinary empty-patch grader keeps this case in the full dataset denominator.
                    write(a.stage/'results/trajectories'/(row['instance_id']+'.json'),
                        dict(info=result, messages=[dict(role='exit', content='BudgetInfeasible',
                             extra=dict(exit_status='BudgetInfeasible', submission=''))],
                             trajectory_format='minisweagent-1',
                             memory=dict(variant=a.method, requests=[], selections=[], goal_changes=[]),
                             usage={}))
                    write(a.stage/(row['instance_id']+'.budget-infeasible.json'), result)
                    return result
                def factory(**kwargs):
                    return create_model(a.method,config,backend,**kwargs)
                result = run_instance(row,rc,a.stage/'results',model_factory=factory)
                result.update(started=start,finished=time.time(),stage=a.stage.name)
                return result
            finally:
                backend.reset()
                LOCAL.case = ''

        with ThreadPoolExecutor(max_workers=config['evaluation']['workers']) as pool:
            futures = [pool.submit(run,row) for row in pending]
            for future in as_completed(futures):
                result = future.result()
                iid = result['instance_id']
                if result['exit_status']=='error':
                    write(a.stage/(iid+'.error.json'),result)
                else:
                    predictions[iid] = result
                    target = a.output/'trajectories'
                    target.mkdir(exist_ok=True)
                    shutil.copy2(a.stage/'results/trajectories'/f'{iid}.json',target/f'{iid}.json')
                    write(prediction_path,[predictions[i] for i in ids if i in predictions])
                print(json.dumps(dict(event='CASE_DONE',id=iid,status=result['exit_status'],
                                      completed=len(predictions),expected=len(ids))),flush=True)
        with httpx.Client(timeout=10) as control:
            response = control.post(root_url+'/benchmark-freeze')
            response.raise_for_status()
            write(a.stage/'measurement-final.json',response.json())
        write(a.stage/'runner-completed.json',dict(time=time.time(),completed=len(predictions),expected=len(ids)))
        if list(a.stage.glob('*.error.json')):
            raise RuntimeError('Infrastructure/request errors require inspection before continuation')
    finally:
        backend.close()


if __name__ == '__main__':
    main()
