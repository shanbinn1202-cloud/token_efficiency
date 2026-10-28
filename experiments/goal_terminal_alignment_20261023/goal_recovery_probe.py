"""Real 16-session GPU lifecycle check before formal measurement starts."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time


def probe(server, stage):
    from goal_client import GoalSessionBackend
    config = server.client_config()
    backend = GoalSessionBackend(config)
    started = time.time()
    def session(index):
        messages = [dict(role='system', content='Read each supplied record. Reply only ACK.'),
                    dict(role='user', content=f'Record owner {index}. Reply ACK.')]
        rows = []
        try:
            for turn in range(3):
                answer = backend.complete(model=config['model']['name'], messages=messages, max_tokens=32)
                rows.append(dict(usage=answer.usage.model_dump(), turn=turn))
                messages.append(answer.choices[0].message.model_dump(exclude_none=True))
                # Four concurrent histories cross the actual compression budget.
                content = ('record alpha beta gamma delta epsilon zeta.\n' *
                           (server.budget//8+256)) if turn == 0 and index < 4 else 'Record complete.'
                messages.append(dict(role='user', content=content+'\nReply ACK.'))
            return dict(session=index, requests=len(rows), rows=rows)
        finally:
            backend.reset()
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            sessions = list(pool.map(session, range(16)))
        runtime = server.output/'main/runtime.jsonl'
        rows = [json.loads(line) for line in runtime.read_text().splitlines()]
        selected = [r for r in rows if (r.get('query') or {}).get('selection', {}).get('reason', 'not_selected') is None]
        if not selected:
            raise RuntimeError('Recovery GPU probe never exercised above-budget KV selection')
        failures = server.output/'main/failures.jsonl'
        if failures.exists() and failures.stat().st_size:
            raise RuntimeError('Recovery GPU probe recorded native request failures')
        result = dict(passed=True, model=server.model, budget=server.budget,
                      sessions=len(sessions), requests=sum(r['requests'] for r in sessions),
                      selected_turns=len(selected), seconds=time.time()-started,
                      excluded_from_benchmark_measurement=True)
        (Path(stage)/'recovery-gpu-probe.json').write_text(json.dumps(result, indent=2))
        return result
    finally:
        backend.close()
