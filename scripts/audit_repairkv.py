"""Compare real GPU RepairKV decisions with the unchanged upstream CPU path."""
import argparse
import json
from pathlib import Path
import torch
from adapters.repairkv.upstream import (score_evicted_positions, select_idlekv_positions,
    build_turn_n_keep_plan, materialize_context_partition)
from phases.phase2_kv_cache.src.kv_utils import inject_kv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-full-pool', action='store_true')
    p.add_argument('--base-device', default='cpu', choices=['cpu','cuda'])
    args = p.parse_args()
    score_evicted_positions._score_on_gpu = False
    results = []
    for path in sorted(args.input.glob('*.pt')):
        data = torch.load(path, map_location='cpu', weights_only=False)
        torch.set_num_threads(data.get('score_cpu_threads', torch.get_num_threads()))
        options = data['options']
        scores = score_evicted_positions(query_rows=data['queries'].float(),
            evicted_cache=data['evicted'], active_cache=data['active'], pooling=options['pooling'])
        selected = select_idlekv_positions(evicted_positions=data['evicted'].positions,
            q2_scores=scores, turn_n_scores=data['turn_n_scores'], k=options['repair_tokens'],
            left=options['burst_left'], right=options['burst_right'])
        error = max(abs(scores[pos]-data['scores'][pos]) for pos in scores)
        result = dict(file=path.name, candidates=len(scores), query_shape=list(data['queries'].shape),
            max_score_absolute_error=error, selection_identical=selected==data['selected'],
            selected_count=len(selected), restored_count=len(data['restored_positions']))
        if data.get('base_answer_tokens') is not None:
            pool = inject_kv(data['active'], data['evicted'], data['evicted'].positions).to_device(args.base_device)
            assert pool.positions == list(range(len(pool)))
            base_plan = build_turn_n_keep_plan(post_q1_cache=pool,
                q1_answer_ids=torch.empty(data['base_answer_tokens'], dtype=torch.long),
                context_len=len(pool), sink_size=options['sink_size'],
                recency_window=options['recency_window'], pooling=options['pooling'])
            base = materialize_context_partition(full_post_q1_cache=pool,
                keep_plan=base_plan, context_budget=len(data['base_positions']))
            result.update(base_selection_identical=base.compressed.positions==data['base_positions'],
                base_score_positions=len(base_plan.importance_scores),
                all_history_scored=set(base_plan.importance_scores)==set(data['turn_n_scores'])==set(pool.positions),
                base_max_score_absolute_error=max(abs(s-data['turn_n_scores'][i])
                    for i,s in base_plan.importance_scores.items()), base_reference_device=args.base_device)
            del pool, base
        elif args.require_full_pool:
            raise ValueError('Capture lacks full-pool base audit metadata')
        results.append(result)
        print(json.dumps(result), flush=True)
        del data
    report = dict(passed=bool(results) and all(r['selection_identical']
        and r.get('base_selection_identical', True) and r.get('all_history_scored', True)
        for r in results), results=results)
    args.output.write_text(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit('RepairKV GPU/CPU selection audit failed')


if __name__ == '__main__':
    main()
