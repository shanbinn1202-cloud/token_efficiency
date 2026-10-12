"""Freeze the reviewed runtime only after all required real-model checks pass."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from tebench.config import ROOT
from scripts.run_async_stage import verify_upstream


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    a=p.parse_args();verify_upstream(a.bundle)
    required=('vanilla-8k-smoke-01','intentkv-8k-smoke-01','repairkv-8k-smoke-01',
              'agentkv-8k-smoke-02','agentkv-8k-preemption-02','repairkv-8k-preemption-02')
    evidence={}
    for label in required:
        stage=a.bundle/label
        complete=json.loads((stage/'completed.json').read_text())
        report=json.loads((stage/'validation.json').read_text())
        launch=json.loads((stage/'launch.json').read_text())
        assert complete['passed'] and report['passed'] and launch['configuration']['backend']['async_scheduling']
        evidence[label]=dict(completed=complete,validation_sha256=hashlib.sha256((stage/'validation.json').read_bytes()).hexdigest())
        if 'preemption' in label:
            assert json.loads((stage/'early-eos.json').read_text())['passed']
    hashes={}
    for base in ('tebench','adapters','configs','engines/vllm/vllm','scripts'):
        for directory,children,files in os.walk(ROOT/base):
            children[:]=[d for d in children if d not in {'.git','__pycache__'}]
            for name in files:
                path=Path(directory)/name
                if path.suffix in {'.py','.yaml','.sh'}:
                    hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    target=a.bundle/'validated_sources.json'
    with target.open('x') as f:
        json.dump(hashes,f,indent=2,sort_keys=True)
    (a.bundle/'validation_summary.json').write_text(json.dumps(dict(passed=True,
        frozen_at=time.time(),source_files=len(hashes),evidence=evidence),indent=2))
    print('ASYNC_IMPLEMENTATION_VALIDATED_AND_FROZEN',len(hashes),flush=True)


if __name__=='__main__':
    main()
