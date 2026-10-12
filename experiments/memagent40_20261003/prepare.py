"""Freeze a balanced, answer-independent 40-QA pilot of the requested dataset."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess

import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path('/u/ssun4/token_efficiency_results/memagent40_20261003')
MODEL = '/work/nvme/bcjw/ssun4/cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'
REVISION = 'a785de0f8d2511a43b9e524b5326e1105a4d218a'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def lossless_chunks(text, tokenizer, chunk_size=4096):
    """Split at native-token character boundaries without dropping any text."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded['offset_mapping']
    starts = [0]
    for i in range(chunk_size, len(offsets), chunk_size):
        offset = offsets[i][0]
        if offset > starts[-1]:
            starts.append(offset)
    starts.append(len(text))
    chunks = [text[a:b] for a, b in zip(starts, starts[1:])]
    assert ''.join(chunks) == text and all(chunks)
    return chunks, len(encoded['input_ids'])


def main():
    BUNDLE.mkdir(parents=True, exist_ok=True)
    if (BUNDLE/'manifest.json').exists():
        raise RuntimeError('Refusing to replace the fixed manifest')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    templates = module(ROOT/'external/MemoryAgentBench/utils/templates.py', 'mab_templates')
    wanted = {
        'ruler_qa1_validation_65536': 5, 'ruler_qa2_validation_65536': 5,
        'icl_banking77_5900shot_balance': 2, 'icl_clinic150_7050shot_balance': 2,
        'icl_nlu_8296shot_balance': 2, 'icl_trec_coarse_6600shot_balance': 2,
        'icl_trec_fine_6400shot_balance': 2,
        'factconsolidation_sh_32k': 3, 'factconsolidation_sh_64k': 2,
        'factconsolidation_mh_32k': 3, 'factconsolidation_mh_64k': 2,
        'infbench_sum_eng_shots2': 10,
    }
    grouped = {key: [] for key in wanted}
    files = sorted((ROOT/'data/memagent_bench/data').glob('*.parquet'))
    for path in files:
        row_index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            for row in batch.to_pylist():
                source = row['source']
                if source in grouped:
                    grouped[source].append((path.name, row_index, row))
                row_index += 1
    rng = random.Random(42)
    selected = []
    exclusions = []
    context_dir = BUNDLE/'contexts'
    context_dir.mkdir(exist_ok=True)
    for source, count in wanted.items():
        records = grouped[source]
        candidates = []
        for filename, row_index, row in records:
            questions, answers = row['questions'] or [''], row['answers']
            if len(questions) > 1:
                assert len(questions) == len(answers)
                pairs = [(i, q, answers[i]) for i, q in enumerate(questions)]
            else:
                pairs = [(0, questions[0], answers)]
            candidates.extend((filename, row_index, row, qi, q, a) for qi, q, a in pairs)
        rng.shuffle(candidates)
        kept = 0
        for filename, row_index, row, qi, question, answer in candidates:
            context_hash = hashlib.sha256(row['context'].encode()).hexdigest()
            path = context_dir/(context_hash+'.json')
            if path.exists():
                ctx = json.loads(path.read_text())
            else:
                chunks, n_tokens = lossless_chunks(row['context'], tokenizer)
                ctx = dict(chunks=chunks, context_tokens=n_tokens, sha256=context_hash)
            # Native context is 262144. Reserve 64K for actual generated turns,
            # tool wrappers and the final answer. Never truncate an input.
            if ctx['context_tokens'] > 196608:
                exclusions.append(dict(source=source, file=filename, row=row_index,
                                       tokens=ctx['context_tokens'], reason='input_above_196608'))
                continue
            if not path.exists():
                path.write_text(json.dumps(ctx, ensure_ascii=False))
            query = templates.get_template(source, 'query', 'Long_context_agent').format(question=question)
            split = filename.split('-0000')[0]
            case_id = f'{source}-r{row_index}-q{qi}'
            selected.append(dict(id=case_id, split=split, source=source, file=filename,
                row=row_index, question_index=qi, question=question, query=query,
                answers=answer, context_file=str(path.relative_to(BUNDLE)),
                context_sha256=context_hash, context_tokens=ctx['context_tokens'],
                chunks=len(ctx['chunks']),
                system=templates.get_template(source, 'system', 'Long_context_agent'),
                metric=('llm_judge_f1' if source.startswith('infbench_sum') else
                        'exact_match' if source.startswith('icl_') else 'substring_exact_match')))
            print(case_id, ctx['context_tokens'], len(ctx['chunks']), flush=True)
            kept += 1
            if kept == count:
                break
        assert kept == count, (source, kept, count)
    assert len(selected) == 40 and len({x['id'] for x in selected}) == 40
    # Interleave skills to keep each allocation's workload representative.
    splits = sorted({x['split'] for x in selected})
    groups = [[x for x in selected if x['split'] == split] for split in splits]
    assert all(len(g) == 10 for g in groups)
    cases = [g[i] for i in range(10) for g in groups]
    manifest = dict(dataset='cxkjntmngm/MemAgent_Bench', revision=REVISION,
        seed=42, count=40, count_unit='independent QA; history replayed from fresh state per QA',
        raw_input_token_eligibility_limit=196608, history_truncation=False,
        model_context_limit=262144, chunk_size_native_tokens=4096,
        protocol='sequential bash read_next history tool; unseen question after last chunk; independent QA',
        sources={name:subprocess.check_output(['git','-C',str(ROOT/'external'/name),'rev-parse','HEAD'],text=True).strip()
                 for name in ('MemoryAgentBench','Mem-alpha')},
        excluded_candidates=exclusions, cases=cases)
    (BUNDLE/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print('Wrote', BUNDLE/'manifest.json', flush=True)


if __name__ == '__main__':
    main()
