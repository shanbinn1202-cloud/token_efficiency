"""One common XML agent/prompt and ARM64 tool environment for all six methods."""
import argparse
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import re

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'archive-reference'
STANDARD = ROOT/'standard-source'
sys.path[:0] = [str(ROOT/'mini242'), str(REFERENCE)]
os.environ.update(MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT='1', AGENTDIET_MODE='off', LITELLM_LOCAL_MODEL_COST_MAP='True',
                  MSWEA_GLOBAL_CONFIG_DIR=str(ROOT/'agent-global-config'))

import httpx
import yaml
from litellm import ModelResponse
from transformers import AutoTokenizer
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
from minisweagent.utils.serialize import recursive_merge
from baselines.common.environment import create_apptainer_environment
from official_compaction import OfficialCompactionPolicy, native_compact

spec = importlib.util.spec_from_file_location('standard_swe_runner', STANDARD / 'examples/swebench/upstream/swebench_runner.py')
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temp.replace(path)


def prepare_worktree(env, row):
    """Reset only the private task checkout; installed packages stay untouched."""
    commit=row['base_commit']
    if not re.fullmatch(r'[0-9a-f]{40}',commit):raise ValueError('Invalid dataset base commit')
    before=env.execute({'command':'git rev-parse HEAD && git diff --stat'})
    result=env.execute({'command':f'git reset --hard {commit} && git diff --exit-code && git diff --cached --exit-code'})
    if result['returncode'] or result.get('exception_info'):
        raise RuntimeError(f'Official task worktree initialization failed: {result}')
    return {'base_commit':commit,'before':before,'reset':result}


class Progress:
    def update_instance_status(self, *args):
        pass


class Model(LitellmTextbasedModel):
    def __init__(self, method, endpoint, tokenizer, output, instance, **kwargs):
        super().__init__(**kwargs)
        self.method, self.endpoint, self.tokenizer = method, endpoint, tokenizer
        self.output, self.instance = output, instance
        self.client = httpx.Client(timeout=3600)
        self.calls = []
        self.kv = None
        if method == 'agentkv':
            from tebench.config import load
            from tebench.backends.kv_session import KVSessionBackend
            c = load()
            from agentkv_settings import OPTIONS
            c['methods']['agentkv']['options'] = dict(OPTIONS)
            c['model'].update(path=tokenizer.name_or_path, enable_thinking=True)
            c['backend']['endpoint'] = endpoint
            c['cache']['kv_budget'] = 8192
            c['generation'].update(seed=42, max_tokens=4096, max_window_size=262144, temperature=0, request_timeout=3600)
            self.kv = KVSessionBackend(c, 'agentkv')
        self.compaction_root=output/'official-compaction'/instance
        self.compactor=None
        if method=='claude_compression':
            self.compaction_root.mkdir(parents=True,exist_ok=True)
            self.compactor=OfficialCompactionPolicy(self.count,self.compact_official)

    def clean(self, messages):
        return [{k: m[k] for k in ('role', 'content', 'reasoning_content') if m.get(k) is not None} for m in messages]

    def format_observation_messages(self, message, outputs, template_vars=None):
        observations = super().format_observation_messages(message, outputs, template_vars)
        if self.method == 'agentkv':
            for observation in observations:
                observation.setdefault('extra', {})['agentkv_tool_observation'] = True
        return observations

    def _prepare_messages_for_api(self, messages):
        prepared = super()._prepare_messages_for_api(messages)
        if self.method == 'agentkv':
            if len(prepared) != len(messages):
                raise ValueError('Cannot align Mini-SWE observation provenance')
            for original, target in zip(messages, prepared):
                if original.get('extra', {}).get('agentkv_tool_observation'):
                    target['extra'] = {'agentkv_tool_observation': True}
        return prepared

    def count(self, messages):
        # Transformers 5 may return a BatchEncoding (len == number of keys),
        # so count the encoded rendered text, exactly as the KV transport does.
        text=self.tokenizer.apply_chat_template(self.clean(messages), tokenize=False, add_generation_prompt=True,
                                                enable_thinking=True, preserve_thinking=True)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def record(self, value):
        self.calls.append(value)
        with (self.output / 'requests.jsonl').open('a') as f:
            f.write(json.dumps({'instance_id': self.instance, **value}, default=str) + '\n')

    def post(self, messages, kind='main', max_tokens=4096, **extra):
        start = time.time()
        body = dict(model='selective-prefix', messages=self.clean(messages), temperature=0, seed=42,
                    max_tokens=max_tokens, stream=False, chat_template_kwargs={'enable_thinking': True})
        if self.method == 'claude_compression':
            body['chat_template_kwargs']['preserve_thinking'] = True
        if kind == 'main':
            body.update(stop=['</mswea_bash_command>'], include_stop_str_in_output=True)
        body.update(extra)
        try:
            res = self.client.post(self.endpoint + '/chat/completions', json=body)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            self.record({'kind':kind,'start':start,'end':time.time(),'error':repr(e)})
            raise
        self.record({'kind':kind,'start':start,'end':time.time(),'usage':data.get('usage',{}),
                     'finish_reason':data['choices'][0]['finish_reason']})
        return ModelResponse(**data)

    def compact_official(self, history):
        started = time.time()
        try:
            continuation,meta=native_compact(self.clean(history),endpoint=self.endpoint,
                output=self.compaction_root/f'compact-{len(self.compactor.events)+1:04d}')
        except Exception as e:
            self.record({'kind':'summary','start':started,'end':time.time(),'error':repr(e),
                         'runtime':'official Claude Code'})
            raise
        self.record({'kind':'summary','start':started,'end':time.time(),'usage':meta['usage'],
                     'runtime':'official Claude Code','metadata':meta})
        return continuation,meta

    def _query(self, messages, **kwargs):
        effective = self.compactor.prepare(messages) if self.compactor else messages
        if self.compactor:
            write(self.output / ('compression-' + self.instance + '.json'), self.compactor.events)
        if self.kv:
            start = time.time()
            try:
                result = self.kv.complete(model='selective-prefix', messages=effective,
                    max_tokens=4096, stop=['</mswea_bash_command>'], include_stop_str_in_output=True)
            except Exception as e:
                self.record({'kind':'main','start':start,'end':time.time(),'error':repr(e)})
                raise
            self.record({'kind':'main','start':start,'end':time.time(),'usage':result.usage.model_dump(),
                         'kv':self.kv.last_measurement})
            return result
        return self.post(effective)

    def close(self):
        if self.kv:
            self.kv.close()
        elif self.method in {'full_context','random','sliding_window','goal'}:
            reply = self.client.post(self.endpoint + '/sessions/close', json={'instance_id':self.instance})
            reply.raise_for_status()
        self.client.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True)
    p.add_argument('--endpoint', default='http://127.0.0.1:19164/v1')
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=64)
    p.add_argument('--steps', type=int, default=120)
    args = p.parse_args()
    if args.method=='claude_compression' and (ROOT/'claude-compression-blocked.json').exists():
        raise RuntimeError('Custom summary withdrawn: official Claude Code compaction integration required')
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = json.loads((ROOT / 'dataset-first64.json').read_text())[:args.count]
    images = json.loads((ROOT / 'images-manifest.json').read_text())['images']
    config = recursive_merge(
        yaml.safe_load((ROOT/'mini242/minisweagent/config/benchmarks/swebench_xml.yaml').read_text()),
        yaml.safe_load((STANDARD/'examples/swebench/upstream/swebench_prompt_override.yaml').read_text()))
    results, times, failures = [], [], []
    lock = threading.Lock()

    def run(row):
        iid = row['instance_id']; started = time.time(); model = env = agent = None
        entry = {'instance_id':iid, 'start':started}
        try:
            env = create_apptainer_environment(images[iid]['image'], executable='apptainer', timeout=120)
            write(args.output/(iid+'.environment.json'),prepare_worktree(env,row))
            mc = dict(config['model']);mc.pop('model_class',None)
            mc.update(model_name='openai/selective-prefix',cost_tracking='ignore_errors',model_kwargs={})
            model = Model(args.method,args.endpoint,tokenizer,args.output,iid,**mc)
            if model.compactor:
                # Official summaries point to their pre-compaction transcript.
                # Expose only this task's compaction records, read-only, at the
                # same path so Mini-SWE's existing Bash tool can read that link.
                env.config.exec_args.extend(['--bind',f'{model.compaction_root}:{model.compaction_root}:ro'])
            ac = dict(config['agent']); ac.update(step_limit=args.steps,cost_limit=0,output_path=args.output/(iid+'.traj.json'))
            agent = upstream.SmartAgent(model, env, progress_manager=Progress(), instance_id=iid, **ac)
            result = agent.run(row['problem_statement'])
            pred = {'instance_id':iid,'model_name_or_path':'Qwen3.6-27B','model_patch':result.get('submission',''),
                    'exit_status':result.get('exit_status','unknown')}
            agent.save(args.output/(iid+'.traj.json'),{'info':{'instance_id':iid}})
            with lock: results.append(pred)
        except Exception:
            entry['error'] = traceback.format_exc()
            with lock: failures.append(entry.copy())
        finally:
            try:
                if model: model.close()
            except Exception:
                entry['close_error'] = traceback.format_exc()
                with lock: failures.append(entry.copy())
            entry['end'] = time.time();entry['e2e_seconds'] = entry['end']-started
            entry['turns'] = agent.n_calls if agent else 0
            entry['main_completion_tokens'] = sum(x.get('usage',{}).get('completion_tokens',0) for x in (model.calls if model else []) if x['kind']=='main')
            entry['auxiliary_completion_tokens'] = sum(x.get('usage',{}).get('completion_tokens',0) for x in (model.calls if model else []) if x['kind']=='summary')
            with lock:
                times.append(entry)
                write(args.output/'predictions.json',results)
                write(args.output/'timings.json',times)
                write(args.output/'failures.json',failures)
            if env: env.cleanup()

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(run,rows))
    span = max(x['end'] for x in times)-min(x['start'] for x in times)
    tokens = sum(x['main_completion_tokens'] for x in times)
    report = {'method':args.method,'expected':len(rows),'completed':len(results),'infrastructure_errors':len(failures),
              'main_completion_tokens':tokens,'auxiliary_completion_tokens':sum(x['auxiliary_completion_tokens'] for x in times),
              'e2e_seconds':span,'output_tokens_per_second':tokens/span,'graded':False,'step_limit':args.steps}
    write(args.output/'metrics.json',report);print(json.dumps(report),flush=True)
    if failures: raise SystemExit(1)


if __name__=='__main__': main()
