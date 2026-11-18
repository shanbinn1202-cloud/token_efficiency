"""The official SWE-bench test specification and grading parser in each local ARM64 sandbox."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import sys
import time
from local_common import dataset,task_env,write


def grade(item):
    row,pred,out,manifest=item
    from baselines.common import harness
    destination=out/'grades'/row['instance_id']
    destination.mkdir(parents=True,exist_ok=True)
    def create(image_name,*,executable='apptainer',**kwargs):
        assert executable=='apptainer'
        return task_env(image_name,row['base_commit'])[0]
    originals=harness.load_dataset,harness.create_apptainer_environment
    harness.load_dataset=lambda name,split:[row]
    harness.create_apptainer_environment=create
    try:
        harness.run_apptainer_harness(predictions=[pred],dataset_name='princeton-nlp/SWE-bench_Verified',
               split='test',instance_ids=[row['instance_id']],max_workers=1,timeout=1800,
               run_id='swe150-'+out.parent.name,report_path=destination/'report.json',executable='apptainer',
               image_manifest=str(manifest))
    finally:
        harness.load_dataset,harness.create_apptainer_environment=originals
    return row['instance_id']


def recover_partial(out):
    spec=json.loads((out/'run-spec.json').read_text())
    predictions=[]
    main_tokens=summary_tokens=0
    for iid in spec['instance_ids']:
        receipt=out/(iid+'.completed.json')
        if receipt.exists():
            pred=json.loads(receipt.read_text())['prediction']
            assert pred['instance_id']==iid
            predictions.append(pred)
        requests=out/(iid+'.requests.jsonl')
        if requests.exists():
            with requests.open() as stream:
                for line in stream:
                    if not line.endswith('\n'):
                        continue
                    rec=json.loads(line)
                    n=rec.get('usage',{}).get('completion_tokens',0)
                    if rec['kind']=='main':
                        main_tokens+=n
                    elif rec['kind']=='summary':
                        summary_tokens+=n
    metrics={'method':spec['method'],'expected':spec['expected'],'completed':len(predictions),
             'e2e_seconds':None,'output_tokens_per_second':None,'throughput_final':False,
             'main_completion_tokens_receipted':main_tokens,'auxiliary_completion_tokens_receipted':summary_tokens,
             'snapshot_time':time.time(),'partial_recovery':True}
    return metrics,predictions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--partial',action='store_true',help='Grade finished tasks from per-task receipts after an interrupted allocation')
    a=p.parse_args()
    rows,_=dataset();out=a.output.resolve();manifest=a.manifest.resolve()
    if a.partial:
        metrics,entries=recover_partial(out)
    else:
        metrics=json.loads((out/'metrics.json').read_text())
        entries=json.loads((out/'predictions.json').read_text())
    metrics['graded']=0
    preds={p['instance_id']:p for p in entries}
    scope={r['instance_id'] for r in rows[:metrics['expected']]}
    assert set(preds)<=scope
    pending=[]
    for row in rows:
        iid=row['instance_id']
        if iid not in preds:continue
        report=out/'grades'/iid/'report.json'
        if report.is_file():
            previous=json.loads(report.read_text())
            if iid in previous.get('submitted_ids',[]) and not previous.get('infrastructure_error_ids'):continue
        pending.append((row,preds[iid],out,manifest))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(grade,work) for work in pending]
        for f in as_completed(futures):print('Graded',f.result(),flush=True)
    evaluated=set();resolved=set();infra=set()
    for iid in preds:
        pth=out/'grades'/iid/'report.json'
        if not pth.is_file():continue
        result=json.loads(pth.read_text())
        assert all(set(result[k])<={iid} for k in ('submitted_ids','resolved_ids','infrastructure_error_ids'))
        failures=set(result['infrastructure_error_ids']);infra.update(failures)
        evaluated.update(set(result['submitted_ids'])-failures);resolved.update(result['resolved_ids'])
    complete=evaluated==scope and not infra
    metrics.update(graded=len(evaluated),resolved=len(resolved),resolved_ids=sorted(resolved),
                   infrastructure_error_ids=sorted(infra),missing_or_ungraded_ids=sorted(scope-evaluated),
                   complete=complete,accuracy_final=complete,accuracy_percent=100*len(resolved)/len(scope) if complete else None)
    write(out/('partial-result.json' if a.partial else 'result.json'),metrics)
    print(json.dumps({k:metrics[k] for k in ['complete','graded','resolved','accuracy_percent','output_tokens_per_second','e2e_seconds']}))
    return int(not complete)


if __name__=='__main__':sys.exit(main())
