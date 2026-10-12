"""Native async reset after AgentKV eviction, followed by a checkpointed turn."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import uuid

import httpx
from transformers import AutoTokenizer
from tebench.config import load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--events',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=load(a.config)
    tokenizer=AutoTokenizer.from_pretrained(c['model']['path'],local_files_only=True)
    def encode(s): return tokenizer.encode(s,add_special_tokens=False)
    messages=[{'role':'user','content':'Remember the identifier agentkv_native_7391.\n'+
        'repository '* (c['cache']['kv_budget']+257)+'\nCount all integers from 1 to 500.'}]
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
    prefix=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=True)
    prompt=encode(text);header=len(encode(text[len(prefix):]))
    common=dict(method='agentkv',budget=c['cache']['kv_budget'],max_tokens=512,temperature=0,
        top_p=1.,top_k=-1,repetition_penalty=1.,seed=0,header_tokens=header,
        prefix_tokens=len(prompt)-header,ignore_eos=True)
    def rows():
        return [json.loads(line) for line in a.events.read_text().splitlines()] if a.events.exists() else []
    def trajectory(forced):
        sid=uuid.uuid4().hex
        with httpx.Client(base_url=c['backend']['endpoint'].rstrip('/')+'/',timeout=1200) as client:
            def post(path,body):
                response=client.post(path,json=body);response.raise_for_status();return response.json()
            try:
                body=dict(common,session_id=sid,input_ids=prompt)
                if forced:
                    before=sum(r['event']=='compacted' for r in rows())
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        future=pool.submit(post,'kv-session/complete',body)
                        deadline=time.monotonic()+1200
                        while sum(r['event']=='compacted' for r in rows())<=before:
                            assert not future.done(), 'Generation ended before the reset'
                            if time.monotonic()>deadline: raise TimeoutError('No compaction event')
                            time.sleep(.05)
                        result=httpx.post(c['backend']['endpoint'].removesuffix('/v1')+
                            '/reset_prefix_cache?reset_running_requests=true',timeout=1200)
                        result.raise_for_status()
                        first=future.result(timeout=1200)
                else:
                    first=post('kv-session/complete',body)
                assert first['native_request_finished'] and first['compactions']>0
                tail=first['output_ids'][-1:]+encode('<|im_end|>\n<|im_start|>user\nContinue counting.<|im_end|>\n<|im_start|>assistant\n<think>\n')
                second=post('kv-session/complete',dict(common,session_id=sid,input_ids=tail,max_tokens=256))
                assert second['cached_tokens']>0 and second['native_request_finished']
                assert second['checkpoint_bytes']>0
                return dict(first=first,second=second)
            finally:
                post('kv-session/close',dict(session_id=sid))
    baseline=trajectory(False); forced=trajectory(True)
    assert forced['first']['recompute']['preemptions']>0
    assert forced['first']['recompute']['replayed_compactions']>0
    a.output.write_text(json.dumps(dict(passed=True,baseline=baseline,forced=forced,
        text_identical=baseline['first']['output_ids']==forced['first']['output_ids']),indent=2))
    print('AGENTKV_ASYNC_NATIVE_PREEMPTION_PASSED',flush=True)


if __name__=='__main__':
    main()
