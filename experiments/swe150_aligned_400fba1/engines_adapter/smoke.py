"""Check the live engine's active async configuration and the method's data path."""
import argparse
import json
from pathlib import Path
import sys
import uuid
import httpx


def check_config(client):
    response=client.get('/server_info',params={'config_format':'json'})
    response.raise_for_status()
    cfg=response.json()['vllm_config']
    expected={'scheduler_config':{'async_scheduling':True,'max_num_batched_tokens':16384,'max_num_seqs':24},
              'cache_config':{'kv_cache_memory_bytes':25*1024**3},
              'model_config':{'max_model_len':262144,'seed':42},
              'additional_config':{'remote_ra_kv_layout':{'page_tokens':16,'mamba_tokens':784,'mamba_slots':49}}}
    for section,values in expected.items():
        for key,value in values.items():
            if cfg[section].get(key)!=value:
                raise ValueError(f'{section}.{key}={cfg[section].get(key)!r}, expected {value!r}')
    return expected


def agent_probe(client, model):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
    enc=lambda text:tok.encode(text, add_special_tokens=False)
    observation=''.join(f'Observation {i}: source line repeated for the forced compaction probe.\n' for i in range(930))
    messages=[{'role':'system','content':'Continue counting.'},{'role':'user','content':observation}]
    text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
    ids=enc(text)
    assert 8192 < len(ids) < 16384,len(ids)
    sid=uuid.uuid4().hex
    body={'session_id':sid,'method':'agentkv','input_ids':ids,'max_tokens':256,'budget':8192,
          'ignore_eos':True,'temperature':0,'seed':42}
    def ask(payload):
        response=client.post('/v1/kv-session/complete',json=payload)
        response.raise_for_status()
        return response.json()
    try:
        a=ask(body)
        assert a['completion_tokens']==256 and a['compactions']==2 and a['retained_kv_tokens']==8192
        marker='AGENTKV_RESUME_MARKER'
        content='The command returned successfully. Continue counting.'
        transcript=tok.apply_chat_template([{'role':'assistant','content':marker},{'role':'user','content':content}],
                     tokenize=False,add_generation_prompt=True,enable_thinking=True)
        tail=transcript.split(marker,1)[1]
        start=tail.index(content)
        pending=a['output_ids'][-1:]
        eos={tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')}
        if pending[0] in eos:pending=[]
        b=ask({**body,'input_ids':pending+enc(tail),'max_tokens':64,'observation_spans':[[start,start+len(content)]]})
        assert b['cached_tokens']==8192 and b['compactions']==2
        assert b['query_phase_counts'][3]>0 and b['native_request_finished']
        keys=['completion_tokens','compactions','retained_kv_tokens','cached_tokens','query_phase_counts','native_request_finished']
        return {'passed':True,'initial_input_tokens':len(ids),'first':{k:a[k] for k in keys},'restored':{k:b[k] for k in keys}}
    finally:
        client.post('/v1/kv-session/close',json={'session_id':sid}).raise_for_status()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local',type=Path,required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--method',choices=['agentkv','claude_compression'],default='agentkv')
    args=p.parse_args()
    with httpx.Client(base_url=f'http://127.0.0.1:{args.port}',timeout=600) as c:
        audit=check_config(c)
        if args.method=='agentkv':result=agent_probe(c,args.local/'model')
        else:
            harness=Path(__file__).resolve().parents[1]/'source/harnesses/perf-swe32'
            sys.path.insert(0,str(harness))
            import stock
            result=stock.keep_probe(f'http://127.0.0.1:{args.port}','selective-prefix',{'stock_model_kwargs':{'extra_body':{'chat_template_kwargs':{'enable_thinking':True}}}})
            assert result['status']=='PASS',result
    args.output.write_text(json.dumps({'passed':True,'method':args.method,'effective_config':audit,'probe':result},indent=2)+'\n')

if __name__=='__main__':main()
