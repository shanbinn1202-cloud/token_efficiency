"""Review a transport-only delta; require a live drain probe in EVERY allocation.

This reuses existing GPU model/attention validation, not a claim that the queued
interactive v48 GPU trial has already run. Model/algorithm/config/task/environment
sources must remain byte-identical to the prior qualification.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

V=Path(__file__).resolve().parent
D=V.parents[1]
sys.path.insert(0,str(D))
from terminal_fingerprints import identity
from terminal_qualification_policy import require_coverage


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old_path=V/'qualification-invalidated-before-revalidation.json'
    old=json.loads(old_path.read_text())
    for name,expected in old['evidence_sha256'].items():
        assert sha(Path(name))==expected, 'Prior evidence changed: '+name
    current=identity()
    changed={name:{'before':old['source_sha256'].get(name),'after':current.get(name)}
             for name in old['source_sha256'].keys()|current.keys()
             if old['source_sha256'].get(name)!=current.get(name)}
    expected={str(D/n) for n in ['terminal_proxy.py','terminal_lifecycle.py','terminal_lifecycle_probe.py',
                                 'run_terminal.py','terminal_stage.py','terminal_fingerprints.py']}
    assert set(changed)==expected, 'Unexpected change outside reviewed HTTP lifecycle delta'
    revision=json.loads((V/'revision.json').read_text())
    for name,entry in revision['changed_files'].items():
        assert sha(D/name)==entry['after'] and sha(V/'before'/name)==entry['before']
    for name,h in revision['added_files'].items():
        assert sha(D/name)==h
    for name,h in revision['regression_evidence_sha256'].items():
        assert sha(V/name)==h
    log=(V/'unit-tests.log').read_text()
    assert 'Ran 16 tests' in log and log.rstrip().endswith('OK'), 'Regression tests not passed'
    for n in ['long-drain-control.json','configure-git-webserver-oracle.json','configure-git-webserver-nop.json','retained-results-audit.json']:
        assert json.loads((V/n).read_text()).get('passed') is True
    assert json.loads((V/'long-drain-control.json').read_text())['drain_seconds']>60
    require_coverage(json.loads((D/'terminal-images/qualification.json').read_text()))
    stage=(D/'terminal_stage.py').read_text()
    assert "if not a.validation or os.environ.get('TERMINAL_LIFECYCLE_PROBE')=='1':" in stage
    assert stage.index('asyncio.run(probe(')<stage.index("r=c.post('http://127.0.0.1:19840/benchmark-begin')")
    runs=json.loads((V/'retained-results-audit.json').read_text())['runs']
    runtime_evidence=[]
    for row in runs:
        root=D/'terminal'/row['model']/row['method'];allocation=root/'allocations'/row['failed_job']
        assert sha(root/'result/predictions.json')==row['predictions_sha256']
        assert sha(root/'result/throughput.json')==row['first_throughput_sha256']
        runtime=json.loads((allocation/'runtime.json').read_text())
        assert runtime['version']=='0.24.0' and runtime['async_scheduling'] and runtime['max_model_len']==262144
        assert set(runtime['attention'].values())=={'TritonAttentionImpl'}
        runtime_evidence.extend([allocation/'runtime.json',allocation/'source-at-start.json',allocation/'launch.json'])
    evidence=[old_path,V/'revision.json',Path(__file__),V/'unit-tests.log',
              *[V/n for n in ['long-drain-control.json','long_drain_control.py','environment_control.py',
                             'configure-git-webserver-oracle.json','configure-git-webserver-nop.json',
                             'retained-results-audit.json']],
              D/'test_terminal_transport.py',D/'test_terminal_lifecycle.py',D/'test_terminal_runner_lifecycle.py',
              *runtime_evidence]
    review={'passed':True,'time':time.time(),'changed_files':changed,
            'model_gpu_evidence_reused':True,'new_interactive_gpu_validation_completed':False,
            'interactive_validation_job':'3462292','mandatory_gpu_drain_probe_before_formal_measurement':True,
            'algorithm_or_config_or_environment_changes':False,
            'scope':'HTTP cleanup ownership, exception isolation, pre-agent startup retry only',
            'evidence_sha256':{str(p):sha(p) for p in evidence}}
    review_path=V/'transport-only-review.json';review_path.write_text(json.dumps(review,indent=2))
    updated=dict(old,time=time.time(),source_sha256=current,
                 evidence_sha256={**old['evidence_sha256'],**review['evidence_sha256'],str(review_path):sha(review_path)},
                 gpu_validation_reused_after_review=True,
                 new_interactive_gpu_validation_completed=False,
                 mandatory_gpu_drain_probe_before_formal_measurement=True,
                 checks=old['checks']+['V48:exact transport-only source delta; GPU model/config sources unchanged',
                                     'V48:16 fault/isolation/retry tests +65s cancelled-request drain',
                                     'V48:real GPU lifecycle probe required in every formal allocation before measurement'])
    target=D/'qualification/terminal-all.json'
    assert not target.exists(), 'Do not overwrite a concurrently generated qualification'
    tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(updated,indent=2));tmp.replace(target)
    print('Qualified reviewed transport-only repair; per-allocation GPU probe remains mandatory',flush=True)


if __name__=='__main__':
    main()
