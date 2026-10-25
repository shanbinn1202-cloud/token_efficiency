"""Apply the established official summary judge to the balanced 32-case run."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

from tebench.config import ROOT
from experiments.memagent20_20261003.judge import REFERENCE_REVISION
from grade_memagent import grade
from experiments.benchmarks32_20261004.telemetry import write


def prepare_references(bundle):
    import pyarrow.parquet as pq
    cases = json.loads((bundle/'memagent/manifest.json').read_text())['cases']
    wanted = {c['context_sha256']: c for c in cases if c['metric']=='llm_judge_f1'}
    assert len(wanted) > 0
    result = {}
    path = ROOT/'data/memoryagentbench_judge_reference/data/Long_Range_Understanding-00000-of-00001.parquet'
    for batch in pq.ParquetFile(path).iter_batches(batch_size=5):
        for row in batch.to_pylist():
            digest = hashlib.sha256(row['context'].encode()).hexdigest()
            if digest not in wanted:
                continue
            case = wanted[digest]
            answers = row['answers']
            while len(answers)==1 and isinstance(answers[0], list):
                answers = answers[0]
            assert answers == case['answers'], case['id']
            keypoints = row['metadata']['keypoints']
            assert keypoints and all(isinstance(k, str) for k in keypoints), case['id']
            result[case['id']] = dict(context_sha256=digest, answers=answers,
                keypoints=keypoints, qa_pair_ids=row['metadata']['qa_pair_ids'])
    assert len(result)==len(wanted), sorted(set(c['id'] for c in wanted.values())-set(result))
    write(bundle/'memagent/judge-references.json', dict(dataset='ai-hyz/MemoryAgentBench',
        revision=REFERENCE_REVISION, match='exact context SHA256 and exact reference summary', references=result))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--env-file', type=Path)
    p.add_argument('--model')
    p.add_argument('--base-url')
    a = p.parse_args()
    references = prepare_references(a.bundle)
    if a.prepare_only:
        print('Verified official keypoints for all selected summaries; no API calls made.')
        return
    from dotenv import load_dotenv
    load_dotenv(a.env_file or a.bundle/'memagent/judge.env')
    model = a.model or os.environ.get('MEMAGENT_JUDGE_MODEL') or 'gpt-4o-2024-05-13'
    base_url = a.base_url or os.environ.get('OPENAI_BASE_URL') or None
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('Configure the authorized GPT-4o judge endpoint before summary grading')
    from openai import OpenAI
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=base_url, timeout=300, max_retries=2)
    spec = importlib.util.spec_from_file_location('official_summary_judge',
        ROOT/'external/MemoryAgentBench/llm_based_eval/summarization_evaluate.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    for path in sorted((a.bundle/'memagent').glob('*/cases/*/result.json')):
        row = json.loads(path.read_text())
        if not row.get('judge_pending'):
            continue
        ref = references[row['id']]
        assert row['context_sha256']==ref['context_sha256'] and row['answers']==ref['answers']
        prompts = dict(
            fluency=official.fluency_prompt_book.format(text=row['prediction'].strip()),
            recall=official.recall_prompt_book.format(summary=row['prediction'].strip(),
                keypoints='\n'.join(f'{i+1}. {k}' for i, k in enumerate(ref['keypoints']))),
            precision=official.precision_prompt_book.format(expert_summary=ref['answers'][0],
                                                            summary=row['prediction'].strip()))
        judged = {}
        for name, prompt in prompts.items():
            cache = path.parent/('judge-'+name+'.json')
            if cache.exists():
                data = json.loads(cache.read_text())
                assert data['model']==model and data['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest()
            else:
                response = client.chat.completions.create(model=model,
                    messages=[{'role':'user', 'content':prompt}], temperature=0.1, max_tokens=4096)
                text = response.choices[0].message.content
                data = dict(model=model, time=time.time(), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    output=text, parsed=official.parse_json(text), usage=response.usage.model_dump())
                write(cache, data)
            if not isinstance(data['parsed'], dict):
                raise ValueError(f'Invalid judge JSON; review {cache}')
            judged[name] = data['parsed']
        f, r, p = judged['fluency'], judged['recall'], judged['precision']
        recall = r['recall']/len(ref['keypoints'])
        precision = p['precision']/p['sentence_count'] if p['sentence_count']>0 else 0
        assert f['fluency'] in (0, 1) and 0<=recall<=1 and 0<=precision<=1
        f1 = f['fluency']*2*recall*precision/(recall+precision) if recall+precision>0 else 0
        row.update(score=f1, judge_pending=False, grading_pending=False,
            judge=dict(model=model, fluency=f['fluency'], recall=recall, precision=precision,
                       f1=f1, reference_revision=REFERENCE_REVISION))
        write(path, row)
        print(json.dumps(dict(id=row['id'], run=path.parents[2].name, f1=f1)), flush=True)
    for output in (a.bundle/'memagent').iterdir():
        if output.is_dir() and (output/'predictions.json').exists():
            grade(a.bundle, output)


if __name__ == '__main__':
    main()
