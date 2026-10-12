"""Real Qwen smoke: long prompt, fresh requests, eight clients, native preemption."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import uuid
import httpx
from tebench.config import load
from tebench.backends import create


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--stats', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    c = load(args.config)
    backend = create(c, 'intentkv')
    unit = backend.tokenizer.encode('All repository tests pass. Check the patch carefully.\n', add_special_tokens=False)
    count = c['cache']['kv_budget']+257
    prompt = (unit*((count+len(unit)-1)//len(unit)))[:count]

    def records():
        if not args.stats.exists():
            return []
        return [json.loads(line) for line in args.stats.read_text().splitlines() if line.strip()]

    def completion(tokens, ids=prompt):
        result = backend.client.completions.create(model=c['model']['name'].split('/')[-1],
            prompt=ids,max_tokens=tokens,temperature=0,top_p=1,seed=0,
            extra_body={'top_k':-1,'repetition_penalty':1.,'ignore_eos':True,
                        'skip_special_tokens':False,'cache_salt':uuid.uuid4().hex})
        assert result.usage.completion_tokens == tokens
        return dict(text=result.choices[0].text,usage=result.usage.model_dump())

    baseline = completion(512)
    initial_selected = sum(r['event']=='selected' for r in records())
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(completion,512)
        deadline = time.monotonic()+1200
        while sum(r['event']=='selected' for r in records()) <= initial_selected:
            assert not future.done(), 'Request finished before preemption was exercised'
            if time.monotonic() > deadline:
                raise TimeoutError('No initial IntentKV selection observed')
            time.sleep(.1)
        result = httpx.post(c['backend']['endpoint'].removesuffix('/v1')+
            '/reset_prefix_cache?reset_running_requests=true',timeout=1200)
        result.raise_for_status()
        forced = future.result(timeout=1200)
    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent = list(pool.map(lambda _:completion(64),range(8)))
    # Ordinary chat transport must also create a new cache for every turn.
    messages = [{'role':'user','content':backend.tokenizer.decode(prompt)+'\nReply OK.'}]
    one = backend.complete(model=c['model']['name'],messages=messages,max_tokens=32)
    messages.extend([one.choices[0].message.model_dump(exclude_none=True),
                     {'role':'user','content':'Reply OK again.'}])
    two = backend.complete(model=c['model']['name'],messages=messages,max_tokens=32)
    # One extra request makes the runner drain the preceding finished IDs.
    completion(1,prompt[:16])
    rows = records()
    completed = [r for r in rows if r['event']=='finished' and r['prompt_tokens']==count]
    assert len(completed) >= 10
    preempted = [r for r in completed if r['preemptions'] > 0]
    assert preempted and any(r['replayed_compactions'] > 0 for r in preempted)
    first_two = completed[:2]
    assert first_two[0]['selection_sha256'] == first_two[1]['selection_sha256']
    for r in completed:
        assert r['selected_tokens'] <= c['cache']['kv_budget']
        assert r['history_events'] == 1
    args.output.write_text(json.dumps(dict(passed=True,baseline=baseline,forced=forced,
        text_identical=baseline['text']==forced['text'],selection_identical=True,
        concurrent_requests=len(concurrent),chat_turns=2,finished=completed,
        effective_config=c),indent=2))
    backend.close()
    print('INTENTKV_MODEL_AND_PREEMPTION_PASSED',flush=True)


if __name__ == '__main__':
    main()
