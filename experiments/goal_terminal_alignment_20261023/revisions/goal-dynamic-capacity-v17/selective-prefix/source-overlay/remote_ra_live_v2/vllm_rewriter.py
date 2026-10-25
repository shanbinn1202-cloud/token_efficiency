"""Concurrent HTTP transport for the unchanged bounded Goal rewrite algorithm."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time
import urllib.request
import uuid

from dram_kv.goals.rewriter import (RewriteCommandError, RewriteResult,
                                    validate_rewrite_candidate)
from dram_kv_scripts.qwen_goal_rewriter import (bounded_model_request,
                                               candidate_schema, parse_exact_candidate)

SYSTEM = (
    "Rewrite a selector-side retrieval Goal as exactly one JSON object "
    "matching the supplied schema. This is not an execution plan. Never "
    "emit root_goal, commands, steps, actions, next_action, reasoning, or "
    "unknown fields. Preserve constraints unless their field is present in "
    "the schema. Prefer concise existing entities and unique array items."
)


def prompt_and_schema(payload, tokenizer):
    schema = candidate_schema(constraints_allowed=(payload.get('trigger') == 'user_request'
                              and payload.get('allow_constraint_updates') is True))
    user = json.dumps({'request': bounded_model_request(payload), 'output_json_schema': schema},
                      ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    text = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # Match upstream tokenizer(rendered), including its add_special_tokens default.
    return list(tokenizer(text)['input_ids']), schema


class VLLMGoalRewriter:
    """Each session owns its HTTP request; vLLM batches independent rewrites."""
    def __init__(self, pin, runtime):
        from transformers import AutoTokenizer
        model = Path(pin['model'])
        if sha256((model/'config.json').read_bytes()).hexdigest() != pin['config_sha256']:
            raise ValueError('Goal rewriter checkpoint differs from startup pin')
        if runtime['backend'] != 'vllm-0.24.0-lm-format-enforcer':
            raise ValueError('Unexpected Goal rewriter backend')
        self.endpoint = runtime['endpoint']
        if not self.endpoint.startswith('http://127.0.0.1:'):
            raise ValueError('Job-local rewriter endpoint required')
        self.pin, self.runtime = dict(pin), dict(runtime)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True,
                                                       trust_remote_code=True)
        self.closed = False

    def close(self):
        # Service lifetime is owned by the allocation worker, outside main timing.
        self.closed = True

    def rewrite(self, request):
        if self.closed:
            raise RewriteCommandError('Goal rewriter closed')
        ident = uuid.uuid4().hex
        started = time.perf_counter()
        begin_wall = time.time()
        tokens, schema = prompt_and_schema(request.to_payload(), self.tokenizer)
        body = dict(model='goal-rewriter', prompt=tokens, max_tokens=self.pin['max_new_tokens'],
                    temperature=0, top_p=1, top_k=-1, repetition_penalty=1,
                    presence_penalty=0, frequency_penalty=0, seed=42,
                    skip_special_tokens=True, stream=False, stop_token_ids=[151645,151643],
                    structured_outputs={'json': schema})
        telemetry = dict(id=ident, started_at=begin_wall, input_tokens=len(tokens),
                         output_tokens=0, backend=self.runtime['backend'],
                         prompt_sha256=sha256(json.dumps(tokens).encode()).hexdigest())
        try:
            req = urllib.request.Request(self.endpoint+'/v1/completions',
                data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req, timeout=self.pin['timeout_seconds']) as response:
                result = json.load(response)
            telemetry['output_tokens'] = result['usage']['completion_tokens']
            telemetry['finish_reason'] = result['choices'][0]['finish_reason']
            raw = parse_exact_candidate(result['choices'][0]['text'])
            candidate = validate_rewrite_candidate(raw, request.current_goal,
                trigger=request.trigger, allow_constraint_updates=request.allow_constraint_updates)
            elapsed = (time.perf_counter()-started)*1000
            telemetry['succeeded'] = True
            return RewriteResult(candidate=candidate, latency_ms=elapsed,
                input_tokens=len(tokens), output_tokens=telemetry['output_tokens'],
                raw_response={'candidate':raw, 'model_cost':0.0})
        except Exception as exc:
            telemetry.update(succeeded=False, error=type(exc).__name__+': '+str(exc)[:1024])
            raise RewriteCommandError(telemetry['error'],
                latency_ms=(time.perf_counter()-started)*1000,
                input_tokens=len(tokens), output_tokens=telemetry['output_tokens']) from exc
        finally:
            telemetry.update(ended_at=time.time(), latency_ms=(time.perf_counter()-started)*1000)
            # One append syscall, no global inference lock. Contains no credentials.
            fd = os.open(self.runtime['telemetry'], os.O_WRONLY|os.O_CREAT|os.O_APPEND, 0o600)
            try: os.write(fd, (json.dumps(telemetry)+'\n').encode())
            finally: os.close(fd)
