"""Select eight complete-history QAs per ability only after the SWE phase."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

from experiments.benchmarks32_20261004.prepare_swe import BUNDLE, METHODS, ROOT
from experiments.benchmarks32_20261004.telemetry import write


def assert_swe_complete():
    for method in METHODS:
        for budget in ('16k', '12k', '8k'):
            if method in {'vanilla', 'agentdiet'} and budget != '16k':
                continue
            path = BUNDLE/'swebench'/(method+'-'+budget)/'accuracy.json'
            result = json.loads(path.read_text())
            assert result['graded']==32 and result['accuracy'] is not None, path


def finish_preparation(data):
    import yaml
    manifest = json.loads((data/'manifest.json').read_text())
    counts = Counter(c['split'] for c in manifest['cases'])
    assert len(manifest['cases'])==32 and len(counts)==4 and set(counts.values())=={8}
    for case in manifest['cases']:
        context = json.loads((data/case['context_file']).read_text())
        assert hashlib.sha256(''.join(context['chunks']).encode()).hexdigest()==case['context_sha256']
    for budget in ('8k',):
        cfg = yaml.safe_load((BUNDLE/(budget+'.yaml')).read_text())
        cfg['dataset'].update(name=manifest['dataset'], split='four_abilities_fixed32', count=32)
        path = data/(budget+'.yaml')
        text = yaml.safe_dump(cfg, sort_keys=False)
        if path.exists():
            assert path.read_text()==text
        else:
            path.write_text(text)
    from experiments.benchmarks32_20261004.judge_memagent import prepare_references
    prepare_references(BUNDLE)
    frozen = json.loads((BUNDLE/'protocol_sources.json').read_text())
    for path in [data/'manifest.json', data/'judge-references.json', *sorted((data/'contexts').glob('*.json')), *sorted(data.glob('*.yaml'))]:
        frozen[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(BUNDLE/'protocol_sources.json', frozen)
    write(data/'prepared.json', dict(ready=True, count=32,
        manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()))
    print(json.dumps(dict(selected=32, counts=dict(counts), sources=manifest['sources'])), flush=True)


def main():
    assert_swe_complete()
    from experiments.benchmarks32_20261004.stage import verify
    verify(BUNDLE)
    import pyarrow.parquet as pq
    import yaml
    from transformers import AutoTokenizer
    from experiments.memagent40_20261003.prepare import lossless_chunks, module, REVISION
    data = BUNDLE/'memagent'
    if (data/'manifest.json').exists():
        # Resume preparation after an interrupted reference/configuration check
        # using the already selected cases, never draw another sample.
        finish_preparation(data)
        return
    data.mkdir(exist_ok=True)
    (data/'contexts').mkdir(exist_ok=True)
    config = yaml.safe_load((BUNDLE/'8k.yaml').read_text())
    tokenizer = AutoTokenizer.from_pretrained(config['model']['path'], local_files_only=True)
    templates = module(ROOT/'external/MemoryAgentBench/utils/templates.py', 'mab_templates')
    grouped = {}
    for path in sorted((ROOT/'data/memagent_bench/data').glob('*.parquet')):
        index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            for row in batch.to_pylist():
                grouped.setdefault(row['source'], []).append((path.name, index, row))
                index += 1
    # Compare a fixed evenly spaced probe of each classification family.
    # Only input token lengths influence family choice, never answers/results.
    probes = {}
    for source, records in sorted(grouped.items()):
        if source.startswith('icl_'):
            indices = sorted({round(i*(len(records)-1)/4) for i in range(5)})
            probes[source] = [len(tokenizer.encode(records[i][2]['context'], add_special_tokens=False)) for i in indices]
    ttl = min(probes, key=lambda name: (statistics.median(probes[name]), name))
    wanted = ['ruler_qa2_validation_65536', 'factconsolidation_sh_32k', ttl, 'infbench_sum_eng_shots2']
    # Select only summaries with a matching official reference and keypoints.
    refs = {}
    reference = ROOT/'data/memoryagentbench_judge_reference/data/Long_Range_Understanding-00000-of-00001.parquet'
    for batch in pq.ParquetFile(reference).iter_batches(batch_size=5):
        for row in batch.to_pylist():
            if row['metadata'].get('keypoints'):
                refs[hashlib.sha256(row['context'].encode()).hexdigest()] = row['answers']
    rng = random.Random(42)
    selected, exclusions, context_cache = [], [], {}
    for source in wanted:
        candidates = []
        for filename, index, row in grouped[source]:
            questions, answers = row['questions'] or [''], row['answers']
            pairs = [(i, q, answers[i]) for i, q in enumerate(questions)] if len(questions)>1 else [(0, questions[0], answers)]
            candidates.extend((filename, index, row, qi, q, answer) for qi, q, answer in pairs)
        if source=='infbench_sum_eng_shots2':
            candidates.sort(key=lambda c: (len(c[2]['context']), c[0], c[1], c[3]))
        else:
            rng.shuffle(candidates)
        kept = 0
        for filename, index, row, qi, question, answer in candidates:
            digest = hashlib.sha256(row['context'].encode()).hexdigest()
            if source=='infbench_sum_eng_shots2':
                reference_answers = refs.get(digest)
                if reference_answers is not None:
                    while len(reference_answers)==1 and isinstance(reference_answers[0], list):
                        reference_answers = reference_answers[0]
                if reference_answers != answer:
                    exclusions.append(dict(source=source, row=index, reason='official_summary_reference_unavailable'))
                    continue
            if digest not in context_cache:
                chunks, n_tokens = lossless_chunks(row['context'], tokenizer, chunk_size=4096)
                context_cache[digest] = dict(chunks=chunks, context_tokens=n_tokens, sha256=digest)
            context = context_cache[digest]
            if context['context_tokens']>196608:
                exclusions.append(dict(source=source, row=index, tokens=context['context_tokens'], reason='full_input_above_196608'))
                continue
            path = data/'contexts'/(digest+'.json')
            if not path.exists():
                write(path, context)
            split = filename.split('-0000')[0]
            selected.append(dict(id=f'{source}-r{index}-q{qi}', split=split, source=source,
                file=filename, row=index, question_index=qi, question=question, answers=answer,
                query=templates.get_template(source, 'query', 'Long_context_agent').format(question=question),
                system=templates.get_template(source, 'system', 'Long_context_agent'),
                context_file=str(path.relative_to(data)), context_sha256=digest,
                context_tokens=context['context_tokens'], chunks=len(context['chunks']),
                metric='llm_judge_f1' if source.startswith('infbench_sum') else 'exact_match' if source.startswith('icl_') else 'substring_exact_match'))
            kept += 1
            if kept==8:
                break
        assert kept==8, (source, kept)
    counts = Counter(c['split'] for c in selected)
    assert len(counts)==4 and set(counts.values())=={8}
    groups = [[c for c in selected if c['split']==split] for split in sorted(counts)]
    cases = [group[i] for i in range(8) for group in groups]
    manifest = dict(dataset='cxkjntmngm/MemAgent_Bench', revision=REVISION,
        created=time.time(), seed=42, count=32, cases=cases, history_truncation=False,
        model_context_limit=262144, raw_input_token_eligibility_limit=196608,
        chunk_size_native_tokens=4096, classification_native_token_probes=probes,
        sampling='8 per ability; seeded QA order for AR/CR/TTL; 8 shortest complete summary histories by character length among eligible records; no model outputs used.',
        sources=wanted, excluded_candidates=exclusions,
        distinct_contexts_by_ability={split:len({c['context_sha256'] for c in cases if c['split']==split}) for split in counts},
        count_unit='independent QA replay; conflict-resolution QAs share the chosen 32K source history',
        protocol='Full chronological history via read_next; question appears only after final chunk; fresh state per QA.')
    write(data/'manifest.json', manifest)
    finish_preparation(data)


if __name__=='__main__':
    main()
