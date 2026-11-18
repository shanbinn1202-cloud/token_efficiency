"""Start the Qwen3.6-27B native vLLM server for the two custom baselines."""
from pathlib import Path
import argparse
import json
import os
import sys
from agentkv_settings import OPTIONS

REFERENCE=Path('/projects/bcjw/ssun4/swe64_standard_20261115/formal-code-20261115/archive-reference')


def command(local, method, port):
    is_agentkv = method == 'agentkv'
    directory = 'baselines' if is_agentkv else 'goal'
    source = local / 'engines' / directory / 'source'
    env = {k:v for k,v in os.environ.items() if not k.startswith(('REMOTE_RA_', 'VLLM_POLICY_RUNTIME', 'VLLM_AGENTKV', 'VLLM_REPAIRKV', 'VLLM_INTENTKV', 'VLLM_NATIVE_WINDOW', 'VLLM_COMPACT_WINDOW', 'VLLM_RANDOM_ATTENTION'))}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               PYTHONPATH=os.pathsep.join([str(source), str(REFERENCE)]),
               VLLM_USE_V2_MODEL_RUNNER='0', VLLM_SERVER_DEV_MODE='1')
    layout={'remote_ra_kv_layout': {'page_tokens':16,'mamba_tokens':784,'mamba_slots':49}}
    cmd=[str(local/'venv/bin/python'),'-B','-m','vllm.entrypoints.openai.api_server',
         '--model',str(local/'model'),'--served-model-name','selective-prefix',
         '--host','127.0.0.1','--port',str(port),'--dtype','bfloat16','--seed','42',
         '--max-model-len','262144','--max-num-batched-tokens','16384',
         '--max-num-seqs','24','--kv-cache-memory-bytes',str(25*1024**3),
         '--async-scheduling','--scheduling-policy','fcfs','--language-model-only',
         '--distributed-executor-backend','uni','--mamba-cache-mode','none' if is_agentkv else 'align',
         '--gdn-prefill-backend','triton','--attention-backend','FLASH_ATTN',
         '--attention-config',json.dumps({'flash_attn_version':3}),
         '--block-size','16','--mamba-block-size','784','--reasoning-parser','qwen3',
         '--enable-auto-tool-choice','--tool-call-parser','qwen3_xml','--chat-template-content-format','string',
         '--compilation-config',json.dumps({'mode':'VLLM_COMPILE','cudagraph_mode':'FULL_AND_PIECEWISE'})]
    if is_agentkv:
        env.update(TEBENCH_VLLM_ROOT=str(source), VLLM_AGENTKV='1',
                   VLLM_AGENTKV_BUDGET='8192', VLLM_AGENTKV_OPTIONS=json.dumps(OPTIONS))
        layout['kv_session_extension']={'method':'agentkv','budget':8192,'options':OPTIONS,'source':'swe150_aligned_400fba1'}
        cmd+=['--no-enable-prefix-caching','--middleware','vllm.tebench.server.middleware',
              '--worker-extension-cls','vllm.tebench.worker.WorkerExtension',
              '--kv-transfer-config',json.dumps({'kv_connector':'AgentKVConnector','kv_role':'kv_both',
                     'kv_connector_module_path':'vllm.tebench.agentkv_connector'})]
    else:
        cmd+=['--enable-prefix-caching','--prefix-caching-hash-algo','sha256',
              '--default-chat-template-kwargs',json.dumps({'preserve_thinking':True,'enable_thinking':True})]
    cmd+=['--additional-config',json.dumps(layout)]
    return cmd,env


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['agentkv','claude_compression'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--local',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    cmd,env=command(args.local,args.method,args.port)
    if args.dry_run:
        print(json.dumps(cmd));sys.exit(0)
    args.output.mkdir(exist_ok=True,parents=True)
    (args.output/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    os.execve(cmd[0],cmd,env)
