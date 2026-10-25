"""Allocation-owned vLLM0.24 auxiliary service; no shared site-package edits."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


class RewriterService:
    def __init__(self, release, model, output, port=18907):
        self.release, self.model, self.output = Path(release), Path(model), Path(output)
        self.port = port
        self.process = None
        self.output.mkdir(exist_ok=False)

    def start(self):
        manifest = json.loads((self.release/'runtime-manifest.json').read_text())
        cfg = json.loads((self.model/'config.json').read_text())
        maxlen = cfg['max_position_embeddings']
        per_token = cfg['num_hidden_layers']*cfg['num_key_value_heads']*cfg.get('head_dim',
                    cfg['hidden_size']//cfg['num_attention_heads'])*4
        # Native maximum context remains available. Auxiliary allocation is separately
        # reported; it does not shrink or borrow the main model's25GiB pool.
        kv_bytes = max(1024**3, ((per_token*maxlen+64*1024**2+1024**3-1)//1024**3)*1024**3)
        args = ['--model',str(self.model),'--served-model-name','goal-rewriter',
            '--host','127.0.0.1','--port',str(self.port),'--dtype','bfloat16',
            '--tokenizer-mode','hf','--distributed-executor-backend','uni',
            '--max-model-len',str(maxlen),'--max-num-seqs','16',
            '--max-num-batched-tokens','4096','--kv-cache-memory-bytes',str(kv_bytes),
            '--gpu-memory-utilization','0.10','--async-scheduling',
            '--scheduling-policy','fcfs','--seed','42','--generation-config','vllm',
            '--structured-outputs-config','{"backend":"lm-format-enforcer"}',
            '--attention-backend','TRITON_ATTN']
        program=("import sys,runpy;sys.path[:0]=[sys.argv[1],sys.argv[2]];"
            "sys.argv=['vllm.entrypoints.openai.api_server']+sys.argv[3:];"
            "runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')")
        command=[sys.executable,'-I','-S','-B','-c',program,str(self.release/'source'),manifest['site'],*args]
        env={k:v for k,v in os.environ.items() if not k.startswith(('REMOTE_RA_','VLLM_POLICY_','SELECTIVE_'))}
        env.update(VLLM_WORKER_MULTIPROC_METHOD='spawn',TOKENIZERS_PARALLELISM='false',
                   VLLM_USE_V2_MODEL_RUNNER='0',PYTHONDONTWRITEBYTECODE='1')
        launch={'command':command,'release':str(self.release),'model':str(self.model),
                'kv_cache_memory_bytes':kv_bytes,'max_model_len':maxlen,
                'max_num_seqs':16,'backend':'vllm-0.24.0-lm-format-enforcer'}
        (self.output/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
        self.log=(self.output/'server.log').open('w')
        self.process=subprocess.Popen(command,env=env,stdout=self.log,stderr=subprocess.STDOUT,
                                      start_new_session=True)
        end=time.monotonic()+900
        while time.monotonic()<end:
            if self.process.poll() is not None:
                raise RuntimeError(f'vLLM rewriter startup exited{self.process.returncode}')
            try:
                with urllib.request.urlopen(self.endpoint+'/health',timeout=2) as response:
                    if response.status==200:
                        (self.output/'ready.json').write_text(json.dumps(launch)+'\n')
                        return self
            except Exception: pass
            time.sleep(2)
        raise TimeoutError('vLLM rewriter startup timeout')

    @property
    def endpoint(self): return f'http://127.0.0.1:{self.port}'

    def bind(self, config, env):
        config['goal_rewriter_runtime']={'backend':'vllm-0.24.0-lm-format-enforcer',
            'endpoint':self.endpoint,'telemetry':str(self.output/'rewrites.jsonl')}
        from hashlib import sha256
        raw=json.dumps(config,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        env['REMOTE_RA_HTTP_CONFIG_SHA256']=sha256(raw).hexdigest()

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid,signal.SIGTERM)
            try:self.process.wait(timeout=35)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid,signal.SIGKILL);self.process.wait()
        if hasattr(self,'log'):self.log.close()
