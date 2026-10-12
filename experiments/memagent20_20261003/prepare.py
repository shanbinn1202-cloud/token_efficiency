"""Reduce the previously fixed sampling order to five QAs per skill.

This reads only the pre-existing dataset manifest and contexts, never any
model predictions or scores. Every selected QA is rerun from a fresh state.
"""
import hashlib
import json
from pathlib import Path
import shutil
import yaml

ROOT = Path(__file__).resolve().parents[2]
OLD = Path('/u/ssun4/token_efficiency_results/memagent40_20261003')
BUNDLE = Path('/u/ssun4/token_efficiency_results/memagent20_20261003')
QUOTAS = {
    'ruler_qa1_validation_65536':3, 'ruler_qa2_validation_65536':2,
    'icl_banking77_5900shot_balance':1, 'icl_clinic150_7050shot_balance':1,
    'icl_nlu_8296shot_balance':1, 'icl_trec_coarse_6600shot_balance':1,
    'icl_trec_fine_6400shot_balance':1,
    'factconsolidation_sh_32k':2, 'factconsolidation_sh_64k':1,
    'factconsolidation_mh_32k':1, 'factconsolidation_mh_64k':1,
    'infbench_sum_eng_shots2':5,
}


def main():
    assert not (BUNDLE/'manifest.json').exists()
    source = OLD/'manifest.json'
    original = json.loads(source.read_text())
    selected = []
    for name,count in QUOTAS.items():
        candidates = [c for c in original['cases'] if c['source']==name]
        assert len(candidates)>=count
        selected.extend(candidates[:count])
    skills = sorted({c['split'] for c in selected})
    groups = [[c for c in selected if c['split']==skill] for skill in skills]
    assert len(selected)==20 and all(len(g)==5 for g in groups)
    cases = [group[i] for i in range(5) for group in groups]
    (BUNDLE/'contexts').mkdir()
    for case in cases:
        src,dst = OLD/case['context_file'],BUNDLE/case['context_file']
        shutil.copy2(src,dst)
        data = json.loads(dst.read_text())
        assert hashlib.sha256(''.join(data['chunks']).encode()).hexdigest()==case['context_sha256']
    manifest = dict(original,count=20,cases=cases,
        sampling='first N in pre-existing seeded per-source sampling order, using fixed source quotas',
        source_quotas=QUOTAS,source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_manifest=str(source),restart=True,reused_predictions=False)
    (BUNDLE/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    for name in ('16k','12k','8k'):
        c = yaml.safe_load((OLD/(name+'.yaml')).read_text())
        c['dataset'].update(count=20,split='four_skills_fixed20')
        assert c['generation']['max_window_size']==262144
        assert c['backend']['kv_cache_memory_bytes']==18*1024**3
        (BUNDLE/(name+'.yaml')).write_text(yaml.safe_dump(c,sort_keys=False))
    print('Prepared 20 fresh QAs; 5 per skill; no predictions reused.')
    for skill,group in zip(skills,groups):
        print(skill,len(group),[c['id'] for c in group])


if __name__ == '__main__':
    main()
