"""Read durable supervision and live per-case progress without touching inference."""
from collections import Counter
import json
from pathlib import Path
import time

B = Path('/u/ssun4/token_efficiency_results/memagent20_20261003')
state = json.loads((B/'supervisor-state.json').read_text())
plan = json.loads((B/'experiment_plan.json').read_text())
selected = {(row['method'], row['budget']):row['count'] for row in plan}
runs = []
for out in sorted((B/'runs').glob('*')):
    method,budget = out.name.rsplit('-',1)
    if (method,budget) not in selected:
        continue
    rows = [json.loads(p.read_text()) for p in (out/'cases').glob('*/result.json')]
    active = []
    for p in sorted((out/'cases').glob('*/progress.json')):
        if (p.parent/'result.json').exists():
            continue
        r = json.loads(p.read_text())
        active.append(dict(id=r['id'],chunks=f'{r["chunks_read"]}/{r["total_chunks"]}',
                           phase=r['stage'],last_update_seconds=round(time.time()-r['time'])))
    runs.append(dict(run=out.name,completed=len(rows),expected=selected[method,budget],
                     statuses=dict(Counter(r['status'] for r in rows)),active=active))
value = dict(time=time.time(),supervisor=state,runs=runs,
             completed_qa=sum(r['completed'] for r in runs),expected_qa=sum(selected.values()))
temporary = B/'live-progress.json.tmp'
temporary.write_text(json.dumps(value,indent=2))
temporary.replace(B/'live-progress.json')
print(json.dumps(value,indent=2))
