"""Compare two or four completed runs from one frozen KV-method budget sweep."""
from copy import deepcopy
import json
from pathlib import Path
import sys

paths=[Path(p).resolve() for p in sys.argv[1:]]
assert len(paths) in (2,4)
runs=[json.loads((p/'run.json').read_text()) for p in paths]
metrics=[json.loads((p/'metrics.json').read_text()) for p in paths]
contexts=[json.loads((p/'final_context_metrics.json').read_text()) for p in paths]
audits=[json.loads((p/'kv_audit.json').read_text()) for p in paths]
assert all(m['eligible_for_comparison'] and m['instances']==20 for m in metrics)
def signature(r):
    c=deepcopy(r['configuration']);c['cache'].pop('kv_budget')
    return (c,r['source_sha256'],r['external'],r['instance_ids'],r['versions'],r['model_files_sha256'])
assert all(signature(r)==signature(runs[0]) for r in runs),'Unaligned source or common configuration'
labels=[('Token sliding' if r['method']=='sliding_window' else 'AgentKV')+' '+str(r['configuration']['cache']['kv_budget']//1024)+'K' for r in runs]
text='# KV methods: same SWE-bench Verified first 20 tasks\n\n'
text+='| Metric | '+' | '.join(labels)+' |\n|---|'+'---:|'*len(paths)+'\n'
for label,key in [('Accuracy','accuracy'),('Mean E2E seconds/task','e2e_latency_seconds_mean'),
    ('Output throughput tokens/s','throughput_output_tokens_per_second'),('Task throughput tasks/hour','throughput_instances_per_hour'),
    ('Average turns','avg_turns'),('Mean request input tokens','avg_context_tokens'),('Batch wall seconds','batch_wall_seconds')]:
    vals=[f"{m[key]:.0%} ({round(m[key]*20)}/20)" if key=='accuracy' else f'{m[key]:,.2f}' for m in metrics]
    text+='| '+label+' | '+' | '.join(vals)+' |\n'
for label,key in [('Mean final input tokens/task','mean_last_prompt_tokens'),('Mean final retained KV tokens/task','mean_final_retained_kv_tokens')]:
    text+='| '+label+' | '+' | '.join(f'{c[key]:,.2f}' for c in contexts)+' |\n'
c=runs[0]["configuration"]
text+=f'\nSame frozen source, model, seed, shared configuration and instance IDs verified. These runs use FA3, retained native GatedDeltaNet state, CUDA Graph and torch.compile; ordinary prefix caching and async scheduling are disabled. Thinking={c["model"]["enable_thinking"]}, workers={c["evaluation"]["workers"]}, max_tokens={c["generation"]["max_tokens"]}, max_turns={c["generation"]["max_turns"]}, max_window_size={c["generation"]["max_window_size"]}, temperature={c["generation"]["temperature"]}.\n'
text+='\nSliding protects the fixed prefix and fills the remaining input budget with the newest token KV, cutting within turns. AgentKV follows the upstream Stage-Q/SnapKV decode-compaction rule; the budget is a target after each 128 generated tokens, not a strict input cap. This difference is part of the methods.\n'
text+='\nOutput throughput = all generated tokens, including thinking, divided by concurrent benchmark batch wall time. E2E includes model, agent and tool work; startup, startup compilation/warmup and grading are excluded; runtime JIT is included. Final input context is the last request input per case. Final retained KV is the actual computed physical KV after the last generation and possible compaction; it excludes the final uncomputed sampled token.\n'
for label,p,a in zip(labels,paths,audits):
    text+=f"\n- {label}: [summary]({p}/summary.md), [per-case statistics]({p}/case_metrics.csv); exits: `{json.dumps(a['exit_status_counts'])}`.\n"
out=paths[0].parent/'comparison.md';out.write_text(text);print(text)
