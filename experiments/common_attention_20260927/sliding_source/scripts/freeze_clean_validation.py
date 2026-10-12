"""Freeze the clean upstream port after its real-model checks complete."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from scripts.run_clean_stage import verify_upstream
from tebench.config import ROOT, load
from tebench.engine import engine_root
from tebench.provenance import snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    args=p.parse_args();b=args.bundle
    verify_upstream(b)
    upstream=json.loads((b/'upstream.json').read_text())
    root=engine_root()
    assert str(root)==upstream['adaptation_checkout']
    assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==upstream['commit']
    assert not subprocess.check_output(['git','-C',upstream['clean_checkout'],'status','--porcelain']).strip()
    patterns=['pristine-vanilla-smoke-*','agentkv-8k-smoke-*','repairkv-8k-smoke-*',
              'intentkv-8k-smoke-*','agentkv-8k-pressure-*','repairkv-8k-pressure-*',
              'vanilla-8k-frontend-*']
    evidence={}
    for pattern in patterns:
        passed=[]
        for stage in b.glob(pattern):
            if not (stage/'completed.json').exists():continue
            complete=json.loads((stage/'completed.json').read_text())
            report=json.loads((stage/'validation.json').read_text())
            if complete['passed'] and report['passed']:
                passed.append((complete['finished'],stage,complete,report))
        assert passed, f'No completed validation: {pattern}'
        _,stage,complete,report=max(passed,key=lambda row:row[0])
        if 'pressure' in pattern:
            assert report['preemptions']>0 and report['recomputed_tokens']>0
            assert json.loads((stage/'early-eos.json').read_text())['passed']
            if pattern.startswith('agentkv'):assert report['replayed_compactions']>0
        evidence[stage.name]=dict(completed=complete,
            validation_sha256=hashlib.sha256((stage/'validation.json').read_bytes()).hexdigest())
    reference=json.loads((b/'vanilla_reference.json').read_text())
    provenance=snapshot(load(b/'8k.yaml'),'vanilla',reference['instance_ids'])
    hashes=provenance['source_sha256']
    for path in (ROOT/'scripts').rglob('*'):
        if path.is_file() and path.suffix in {'.py','.sh'}:
            hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for artifact in json.loads((b/'wheel_artifacts.json').read_text()):
        assert hashlib.sha256((root/artifact['file']).read_bytes()).hexdigest()==artifact['sha256']
    with (b/'validated_sources.json').open('x') as handle:
        json.dump(hashes,handle,indent=2,sort_keys=True)
    patch=subprocess.check_output(['git','-C',str(root),'diff','--binary','HEAD'])
    (b/'vllm-adaptation.patch').write_bytes(patch)
    (b/'validation_summary.json').write_text(json.dumps(dict(passed=True,frozen_at=time.time(),
        engine=provenance['engine'],source_files=len(hashes),evidence=evidence,
        patch_sha256=hashlib.sha256(patch).hexdigest()),indent=2))
    print('CLEAN_UPSTREAM_ASYNC_IMPLEMENTATION_VALIDATED_AND_FROZEN',len(hashes),flush=True)


if __name__=='__main__':
    main()
