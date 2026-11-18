"""The Goal rewriter's vLLM child (``serve.py --goal-rewriter-engine vllm``; parent: goal_rewriter_vllm).

Started by the parent with only the clean environment's site-packages on ``sys.path``
(``-I -S``; not the overlay, whose patched ``vllm`` and runtime gates would load) and an
environment without the runtime's variables. Of the overlay it loads, by file path, only
the vendored upstream rewriter script (``dram_kv_scripts/qwen_goal_rewriter.py``) and the
batched child (``goal_rewriter_child.py``, standard library only) for the prompt, the
schema, the parse and the answer objects.

Per request, as upstream: the envelope checks; the prompt (the upstream
``QwenGoalGenerator.generate`` up to its model call, ``goal_rewriter_child.prepare``:
the system text, the canonical user JSON of the bounded request and the schema, the chat
template without thinking, the Hugging Face tokenizer's ids, so ``input_tokens`` is the
Hugging Face child's count); the JSON schema that generate builds; greedy decoding with
the pinned max_new_tokens; the decode of the generated ids with
``skip_special_tokens=True``; ``parse_exact_candidate`` and ``GenerationFailed`` with its
telemetry; the answer and error objects. A reply cut at max_new_tokens (finish reason
"length") does not parse and answers as upstream, with its token counts.

What changes: the model call. The prompt ids go to one vLLM ``AsyncLLM`` engine
(``TokensPrompt``) with temperature 0, ``max_tokens`` = max_new_tokens, ``stop_token_ids``
= where Transformers stops (the checkpoint's generation_config eos ids) with the
tokenizer's EOS and ``<|im_end|>`` (the same ids for Qwen3-0.6B), the schema as vLLM's
structured output (the backend pinned in the engine arguments; the xgrammar arm drops
uniqueItems, which xgrammar refuses), no detokenization and the final output only.
``generation_latency_ms`` runs from the submission to the engine to its final output, so it
includes the time the request shared the engine's batch with others.

lm-format-enforcer on vLLM 0.24: vLLM's backend loads ``lmformatenforcer.integrations.vllm``
for its tokenizer data, and lm-format-enforcer 0.11.3's module imports
``vllm.transformers_utils.tokenizer``, which vLLM 0.24 no longer has (its tokenizers moved
to ``vllm.tokenizers``): the backend never starts and every request fails (job rw2).
This script puts its own module under that name in ``sys.modules`` before vLLM loads
(here, and in the EngineCore: a forked one inherits it, a spawned one runs this script's
top level as ``__mp_main__``). Its one function builds the Hugging Face child's tokenizer
data (lm-format-enforcer's Transformers builder over the tokenizer's ids, as
``build_transformers_prefix_allowed_tokens_fn`` builds it), sized for vLLM's bitmask of
the model's vocabulary: the ids past the tokenizer (padding rows; they decode to "", which
lm-format-enforcer's own vLLM builder would allow at every step) are never allowed, as in
the Hugging Face child. The process-local Transformers alias is the upstream script's.

Startup check: once the engine is up and before the readiness record, one guided request
(a schema whose only value an unguided reply would not produce) must answer that value,
end at a stop token and use only the tokenizer's ids; otherwise the child ends without
the readiness record, so the server does not start (startup only, nothing per request).

Protocol: the upstream readiness record once the engine is up, then the upstream line
protocol with several requests in flight: one answer line per request, written as it
completes (under a lock), in any order; ``{"abort": request_id}`` aborts that request in
the engine and it gets no answer. stdout carries protocol lines only: the protocol writes
to a duplicate of fd 1, and fd 1 itself points at fd 2 before vLLM is imported, so vLLM's
logger, its EngineCore and any C-level print go to stderr. The end of stdin shuts the
engine down and ends the child. An engine that died ends the child (its parent fails the
calls in flight and starts a new child on the next call).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from time import perf_counter
from types import ModuleType, SimpleNamespace

HERE = Path(__file__).resolve().parent
_CHILD = 'remote_ra_goal_rewriter_child'    # goal_rewriter_child.py, loaded by file path
ABORT = 'abort'
IM_END = '<|im_end|>'
ENGINE = 'vllm'
PENDING = object()                          # submitted, its future not stored yet
SHUTDOWN_SECONDS = 30.0
LFE_VLLM = 'lmformatenforcer.integrations.vllm'     # vLLM's lm-format-enforcer backend loads this name
_MARK = '_remote_ra_goal_rewriter'                  # this script's module under that name
# The startup check: a schema whose only value an unguided reply would not produce
# (uniqueItems kept for lm-format-enforcer, dropped for xgrammar, as in every request).
PROBE_ID = 'remote-ra-guided-probe'
PROBE_PROMPT = 'Say hello.'
PROBE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["probe", "items"],
                "properties": {"probe": {"type": "string", "enum": [PROBE_ID]},
                               "items": {"type": "array", "items": {"type": "string", "enum": ["a"]},
                                         "minItems": 1, "maxItems": 1, "uniqueItems": True}}}
PROBE_ANSWER = {"probe": PROBE_ID, "items": ["a"]}
PROBE_TOKENS = 128
PROBE_SECONDS = 120.0


def log(message):
    sys.stderr.write('goal rewriter vllm child: %s\n' % message)
    sys.stderr.flush()


def batch_child():
    """goal_rewriter_child.py as a module, by file path (its upstream loader, prepare and
    error_answer)."""
    module = sys.modules.get(_CHILD)
    if module is None:
        spec = importlib.util.spec_from_file_location(_CHILD, HERE / 'goal_rewriter_child.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[_CHILD] = module
        spec.loader.exec_module(module)
    return module


# ---- one request: the upstream generate up to its model call --------------------------------

class _NoCuda:
    @staticmethod
    def is_available():
        return False


class HostTorch:
    """What the upstream generate uses of torch before its model call, without CUDA: the
    child itself never touches the GPU (the engine does, in its own process)."""
    cuda = _NoCuda()
    inference_mode = staticmethod(contextlib.nullcontext)


class _Schema:
    """The upstream generate's ``JsonSchemaParser(schema)``: here it keeps the schema."""

    def __init__(self, schema):
        self.schema = schema


class Prompts:
    """A request's prompt token ids and JSON schema, as the upstream ``QwenGoalGenerator.generate``
    builds them before its model call (``goal_rewriter_child.prepare`` on a generator with
    ``tokenizer`` and no model); an error the upstream code raises before the model call
    propagates."""

    def __init__(self, tokenizer, max_new_tokens):
        self.C = batch_child()
        U = self.C.upstream()
        generator = U.QwenGoalGenerator.__new__(U.QwenGoalGenerator)
        generator.torch, generator.tokenizer, generator.device = HostTorch(), tokenizer, 'cpu'
        generator.max_new_tokens, generator.model = max_new_tokens, None
        generator.JsonSchemaParser = _Schema
        generator.build_prefix_fn = lambda tokenizer_, parser: parser
        self.generator, self.max_new_tokens = generator, max_new_tokens

    def __call__(self, request):
        arguments = self.C.prepare(self.generator, request)
        rows = arguments['input_ids'].tolist()
        schema = arguments.get('prefix_allowed_tokens_fn')
        if (len(rows) != 1 or arguments.get('do_sample') is not False or
                arguments.get('max_new_tokens') != self.max_new_tokens or not isinstance(schema, _Schema)):
            raise RuntimeError('the upstream generate calls its model otherwise than the vLLM child expects')
        return [int(token) for token in rows[0]], schema.schema


def without_unique_items(value):
    """The schema without its uniqueItems keywords (the xgrammar arm)."""
    if isinstance(value, dict):
        return {key: without_unique_items(item) for key, item in value.items() if key != 'uniqueItems'}
    if isinstance(value, list):
        return [without_unique_items(item) for item in value]
    return value


def stop_token_ids(model, tokenizer):
    """Where the greedy decode stops: the checkpoint's generation_config eos ids (where
    Transformers' generate stops), the tokenizer's EOS and <|im_end|>."""
    ids = set()
    try:
        from transformers import GenerationConfig
        value = GenerationConfig.from_pretrained(model, local_files_only=True).eos_token_id
    except Exception as error:
        log('no generation_config eos ids (%s: %s)' % (type(error).__name__, error))
        value = None
    ids.update(int(v) for v in (value if isinstance(value, (list, tuple)) else () if value is None else (value,)))
    eos = getattr(tokenizer, 'eos_token_id', None)
    if type(eos) is int:
        ids.add(eos)
    try:
        im_end = tokenizer.convert_tokens_to_ids(IM_END)
    except Exception:
        im_end = None
    if type(im_end) is int and im_end != getattr(tokenizer, 'unk_token_id', None):
        ids.add(im_end)
    if not ids:
        raise RuntimeError('the Goal rewriter checkpoint names no stop token')
    return sorted(ids)


# ---- lm-format-enforcer's tokenizer data for vLLM's backend -------------------------------------

def transformers_alias():
    """The upstream script's process-local alias (``QwenGoalGenerator``): lm-format-enforcer
    <= 0.11's Transformers integration imports PreTrainedTokenizerBase from the pre-v5 module."""
    try:
        from transformers.tokenization_utils import PreTrainedTokenizerBase  # noqa: F401
    except ImportError:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        import transformers.tokenization_utils as tokenization_utils
        tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase


def enforcer_tokenizer_data(tokenizer, use_bitmask=False, vocab_size=None):
    """vLLM's ``build_vllm_token_enforcer_tokenizer_data(tokenizer, use_bitmask=True,
    vocab_size=<the model's>)``: the Hugging Face child's data (``build_token_enforcer_tokenizer_data``
    over ``len(tokenizer)`` ids: the same tokens, strings, decoder and EOS) with the bitmask
    width ``vocab_size``. The tokenizer is unwrapped as lm-format-enforcer's vLLM builder does."""
    transformers_alias()
    from lmformatenforcer import TokenEnforcerTokenizerData
    from lmformatenforcer.integrations.transformers import _build_regular_tokens_list, _decode_function
    from transformers import PreTrainedTokenizerBase
    if hasattr(tokenizer, 'get_tokenizer'):
        tokenizer = tokenizer.get_tokenizer()
    if hasattr(tokenizer, 'tokenizer'):
        tokenizer = tokenizer.tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError('the Goal rewriter\'s lm-format-enforcer needs a Hugging Face tokenizer, not %s'
                        % type(tokenizer).__name__)
    size = len(tokenizer)
    width = size if vocab_size is None else vocab_size
    if type(width) is not int or width < size:
        raise ValueError('the bitmask width %r is below the tokenizer\'s %d ids' % (width, size))
    return TokenEnforcerTokenizerData(_build_regular_tokens_list(tokenizer, size),
                                      functools.partial(_decode_function, tokenizer), tokenizer.eos_token_id,
                                      use_bitmask, width)


def _enforcer_attribute(name):
    return getattr(importlib.import_module('lmformatenforcer'), name)


def install_enforcer():
    """This script's module under ``LFE_VLLM`` in ``sys.modules`` (before vLLM loads it)."""
    present = sys.modules.get(LFE_VLLM)
    if present is not None and getattr(present, _MARK, False) is not True:
        raise SystemExit('the vLLM Goal rewriter child found %s loaded before it' % LFE_VLLM)
    module = ModuleType(LFE_VLLM, 'lm-format-enforcer\'s tokenizer data for vLLM 0.24 (goal_rewriter_vllm_child).')
    module.build_vllm_token_enforcer_tokenizer_data = enforcer_tokenizer_data
    module.__getattr__ = _enforcer_attribute        # TokenEnforcerTokenizerData and the rest: the package's
    setattr(module, _MARK, True)
    sys.modules[LFE_VLLM] = module


# ---- the engine ------------------------------------------------------------------------------

def vllm_api():
    """The clean vLLM's classes (imported here, after stdout was moved to stderr)."""
    from vllm import SamplingParams
    from vllm.config import CompilationConfig, StructuredOutputsConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind, StructuredOutputsParams
    from vllm.v1.engine.async_llm import AsyncLLM
    import vllm
    if Path(vllm.__file__).resolve().is_relative_to(HERE.parent):
        raise SystemExit('the vLLM Goal rewriter child loaded the overlay\'s vllm')
    return SimpleNamespace(SamplingParams=SamplingParams, CompilationConfig=CompilationConfig,
                           StructuredOutputsConfig=StructuredOutputsConfig, AsyncEngineArgs=AsyncEngineArgs,
                           TokensPrompt=TokensPrompt, RequestOutputKind=RequestOutputKind,
                           StructuredOutputsParams=StructuredOutputsParams, AsyncLLM=AsyncLLM)


async def build_engine(api, values):
    """The engine of the pinned arguments (on the loop thread)."""
    arguments = dict(values)
    arguments['compilation_config'] = api.CompilationConfig(**arguments['compilation_config'])
    arguments['structured_outputs_config'] = api.StructuredOutputsConfig(**arguments['structured_outputs_config'])
    return api.AsyncLLM.from_engine_args(api.AsyncEngineArgs(**arguments))


class Loop:
    """One asyncio loop on a dedicated thread; it owns the engine."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name='goal-rewriter-vllm-loop', daemon=True)
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coroutine, timeout=None):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout)


class Lines:
    """The protocol stream: one canonical line per object (the upstream ``_write_json``)."""

    def __init__(self, stream):
        self.stream, self.lock = stream, threading.Lock()

    def __call__(self, value):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n'
        with self.lock:
            self.stream.write(text)
            self.stream.flush()


class Server:
    """The line protocol over one engine: the reader (``serve``, the main thread) checks and
    prepares each request and submits it to the loop; each request answers on completion."""

    def __init__(self, engine, loop, api, tokenizer, prompts, stops, unique_items, write):
        self.engine, self.loop, self.api, self.tokenizer, self.prompts = engine, loop, api, tokenizer, prompts
        self.stops, self.unique_items, self.write = list(stops), unique_items, write
        self.C = batch_child()
        self.U = self.C.upstream()
        self.lock = threading.Lock()            # the requests in flight
        self.tokens = threading.Lock()          # one fast tokenizer, two threads
        self.inflight = {}

    def serve(self, lines):
        for line in lines:
            self.handle(line)

    def handle(self, line):
        """One line: an abort, or an upstream envelope (its errors before the model call
        answered at once, as upstream)."""
        try:
            envelope = json.loads(line)
        except Exception as error:
            self.write(self.C.error_answer('', error))
            return
        if isinstance(envelope, dict) and set(envelope) == {ABORT}:
            self.abort(envelope[ABORT])
            return
        request_id = ''
        try:
            if not isinstance(envelope, dict) or set(envelope) != {"request_id", "request"}:
                raise ValueError("request envelope has unknown or missing fields")
            request_id = envelope["request_id"]
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("request_id must be a non-empty string")
            with self.lock:
                busy = request_id in self.inflight
            if busy:
                request_id = ''                 # this answer must not reach the call in flight
                raise ValueError('request_id is already in flight')
            with self.tokens:
                ids, schema = self.prompts(envelope["request"])
            if not self.unique_items:
                schema = without_unique_items(schema)
        except Exception as error:
            self.write(self.C.error_answer(request_id, error))
            return
        self.submit(request_id, ids, schema)

    def submit(self, request_id, ids, schema):
        with self.lock:
            self.inflight[request_id] = PENDING
        future = asyncio.run_coroutine_threadsafe(self.generate(request_id, ids, schema), self.loop)
        with self.lock:
            if self.inflight.get(request_id) is PENDING:
                self.inflight[request_id] = future

    def sampling(self, schema, max_tokens=None):
        api = self.api
        return api.SamplingParams(temperature=0.0, max_tokens=max_tokens or self.prompts.max_new_tokens,
                                  stop_token_ids=list(self.stops),
                                  structured_outputs=api.StructuredOutputsParams(json=schema), detokenize=False,
                                  output_kind=api.RequestOutputKind.FINAL_ONLY)

    def self_test(self):
        """Startup only: one guided request (``PROBE_SCHEMA``, greedy, the stop ids) must answer
        ``PROBE_ANSWER``, end at a stop token and use only the tokenizer's ids. Its error otherwise."""
        schema = PROBE_SCHEMA if self.unique_items else without_unique_items(PROBE_SCHEMA)
        with self.tokens:
            rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": PROBE_PROMPT}], tokenize=False,
                                                          add_generation_prompt=True, enable_thinking=False)
            ids = [int(token) for token in self.tokenizer(rendered)['input_ids']]
            size = len(self.tokenizer)

        async def run():
            final = None
            async for output in self.engine.generate(self.api.TokensPrompt(prompt_token_ids=ids),
                                                     self.sampling(schema, PROBE_TOKENS), PROBE_ID):
                final = output
            return final
        started = perf_counter()
        final = asyncio.run_coroutine_threadsafe(run(), self.loop).result(PROBE_SECONDS)
        if final is None or not final.outputs:
            raise RuntimeError('the engine ended the request without an output')
        tokens, reason = [int(t) for t in final.outputs[0].token_ids], final.outputs[0].finish_reason
        with self.tokens:
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        if reason != 'stop' or not tokens or tokens[-1] not in self.stops:
            raise RuntimeError('the reply did not end at a stop token (%s): %r' % (reason, text[:200]))
        if max(tokens) >= size:
            raise RuntimeError('the reply holds ids past the tokenizer\'s %d: %s' % (size, [t for t in tokens if t >= size]))
        try:
            value = json.loads(text)
        except ValueError:
            value = None
        if value != PROBE_ANSWER:
            raise RuntimeError('the reply is not the schema\'s only value: %r' % text[:200])
        return len(tokens), (perf_counter() - started) * 1_000

    async def generate(self, request_id, ids, schema):
        try:
            params, prompt = self.sampling(schema), self.api.TokensPrompt(prompt_token_ids=ids)
            started, final = perf_counter(), None
            async for output in self.engine.generate(prompt, params, request_id):
                final = output
            latency_ms = (perf_counter() - started) * 1_000
            if final is None or not final.outputs:
                raise RuntimeError('the engine ended the request without an output')
            answer = self.finish(request_id, [int(t) for t in final.outputs[0].token_ids], len(ids), latency_ms)
        except asyncio.CancelledError:
            log('request %s aborted' % request_id)      # vLLM aborted it in the engine
            raise
        except Exception as error:
            answer = self.C.error_answer(request_id, error)
            if getattr(self.engine, 'errored', False):
                self.write(answer)
                self.fatal(error)
        finally:
            with self.lock:
                self.inflight.pop(request_id, None)
        self.write(answer)

    def finish(self, request_id, tokens, input_tokens, generation_latency_ms):
        """The upstream generate's end: decode, telemetry, parse; the answer object."""
        with self.tokens:
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        telemetry = {"input_tokens": int(input_tokens), "output_tokens": len(tokens),
                     "generation_latency_ms": generation_latency_ms}
        try:
            candidate = self.U.parse_exact_candidate(text)
        except Exception as error:
            failed = self.U.GenerationFailed(f"{type(error).__name__}: {error}", telemetry)
            return self.C.error_answer(request_id, failed)
        return {"request_id": request_id, "result": {"candidate": candidate, "model_cost": 0.0, **telemetry}}

    def abort(self, request_id):
        """Cancel the request's task: vLLM's generate aborts it in the engine. No answer."""
        with self.lock:
            future = self.inflight.pop(request_id, None) if type(request_id) is str else None
        if future is not None and future is not PENDING:
            future.cancel()

    @staticmethod
    def fatal(error):
        log('the engine failed (%s: %s); the child ends' % (type(error).__name__, error))
        os._exit(1)

    def close(self):
        """The end of stdin: cancel what is in flight, shut the engine down."""
        with self.lock:
            pending = [f for f in self.inflight.values() if f is not PENDING]
            self.inflight.clear()
        for future in pending:
            future.cancel()

        async def shutdown():
            self.engine.shutdown()
        try:
            asyncio.run_coroutine_threadsafe(shutdown(), self.loop).result(SHUTDOWN_SECONDS)
        except Exception as error:
            log('engine shutdown failed (%s: %s)' % (type(error).__name__, error))


# ---- startup -----------------------------------------------------------------------------------

def protocol_stream():
    """The protocol's stream: a duplicate of fd 1. fd 1 itself (sys.stdout, vLLM's logger,
    its EngineCore, any C-level print) goes to fd 2 from now on."""
    sys.stdout.flush()
    fd = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(fd, 'w', encoding='utf-8', newline='\n', buffering=1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='goal_rewriter_vllm_child')
    parser.add_argument('--model', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=384)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--engine', required=True, help='the pinned goal_rewriter_engine record (JSON)')
    return parser.parse_args(argv)


def isolated():
    """The overlay (its patched vllm) is not on sys.path."""
    source = HERE.parent
    for entry in sys.path:
        if entry and Path(entry).resolve() == source:
            raise SystemExit('the vLLM Goal rewriter child must not see the runtime overlay on sys.path')


def main(argv=None):
    write = Lines(protocol_stream())
    isolated()
    args = parse_args(argv)
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not args.serve:
        raise SystemExit('the vLLM child serves the NDJSON protocol only (--serve)')
    value = json.loads(args.engine)
    if value.get('engine') != ENGINE or value['engine_args']['model'] != args.model:
        raise SystemExit('the vLLM child needs the goal_rewriter_engine record of its checkpoint')
    sys.stdin.reconfigure(encoding='utf-8')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only,
                                              trust_remote_code=True)
    prompts = Prompts(tokenizer, args.max_new_tokens)
    stops = stop_token_ids(args.model, tokenizer)
    api = vllm_api()
    loop = Loop()
    engine = loop.run(build_engine(api, value['engine_args']))
    server = Server(engine, loop.loop, api, tokenizer, prompts, stops, value['schema_unique_items'], write)
    try:
        probe = server.self_test()
    except BaseException as error:
        log('refused: the guided decoding (%s) failed its startup check: %s: %s' % (
            value['guided_backend'], type(error).__name__, error))
        server.close()
        sys.stderr.flush()
        os._exit(1)
    log('startup check passed: the guided reply in %d tokens, %.0f ms' % probe)
    log('engine ready: structured outputs %s%s, stop token ids %s, batch invariant %s' % (
        value['guided_backend'], '' if value['schema_unique_items'] else ' (schema without uniqueItems)', stops,
        os.environ.get('VLLM_BATCH_INVARIANT', '0') == '1'))
    U = batch_child().upstream()
    write({"protocol": U.PROTOCOL, "version": U.PROTOCOL_VERSION, "ready": True})
    server.serve(sys.stdin)
    server.close()
    sys.stderr.flush()
    os._exit(0)


if __name__ in ('__main__', '__mp_main__'):     # the child; a spawned EngineCore re-runs this top level
    install_enforcer()
if __name__ == '__main__':
    main()
