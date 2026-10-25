"""The batched Goal rewriter child (``serve.py --goal-rewriter-batch N``; parent: goal_rewriter_batch).

Started like the upstream ``dram_kv_scripts/qwen_goal_rewriter.py --serve`` (the same
isolated command and model arguments) with ``--batch N`` added. It loads the model
through the upstream ``QwenGoalGenerator``, prints the upstream v1 readiness record and
serves the upstream NDJSON protocol with one addition: a line ``{"batch": [envelope,
...]}`` (1 to N upstream envelopes) is answered by one line ``{"batch": [answer, ...]}``,
the upstream answer object of each envelope, in order. A plain upstream envelope line is
answered as upstream, by its own answer line.

What stays upstream, per request: the envelope checks, the prompt (system text, the
canonical JSON of the bounded request and the schema, the chat template without
thinking), the JSON schema and its lm-format-enforcer parser and prefix function, greedy
decoding with the pinned max_new_tokens and pad = eos, the decode of the generated
suffix, ``parse_exact_candidate`` and ``GenerationFailed``, the answer and error objects.
``prepare`` runs the upstream ``QwenGoalGenerator.generate`` itself on a shallow copy
whose model only records the arguments of its ``generate`` call; everything before that
call is the upstream code.

What changes: the model call. ``generate_batch`` runs the prepared requests as one
``model.generate``: the prompts left-padded with the pad token and masked, the same
greedy settings, and one prefix function that hands row i, its padding stripped, to
request i's own upstream prefix function, so that each enforcer sees exactly the token
sequence it would see alone. A row that has emitted a stop token (the model's
generation_config eos ids, where Transformers stops a row) only gets the pad token and
its enforcer is not called again. Afterwards each row's suffix is cut after its first stop
token (the sequence Transformers returns for that request alone) and finished as the
upstream generate finishes: decode, telemetry (its own prompt and generated token counts;
``generation_latency_ms`` is the batch's), parse. Batched greedy decoding is not
bit-identical to decoding each request alone: padding and the batch dimension change the
order of floating-point reductions, so a near-tie can flip and a request's candidate (or
its failure) can differ from the unbatched one.

Fallbacks stay per request. A batch of one runs the upstream ``generate`` unchanged. An
out-of-memory error in a batched ``generate`` empties the CUDA cache and runs the batch
in halves (logged to stderr); any other exception in a batched ``generate`` runs each of
its requests alone through the upstream ``generate``, so its answer or error is the
upstream one.

Tokenizer data (``cache_tokenizer_data``, startup only). The upstream generate builds its
prefix function from the tokenizer, which makes lm-format-enforcer rebuild its token table
and prefix tree over the whole vocabulary on every request. The batched child builds that
tokenizer data once and passes it instead of the tokenizer; the parser, the enforcer and
the allowed tokens are the same. A startup self-check compares a fresh and a cached prefix
function on one probe sequence; if anything fails, the child logs why and keeps the
upstream per-request build.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

BATCH = 'batch'
LIMIT = 16                      # serve.py --goal-rewriter-batch N: 2 <= N <= 16
_UPSTREAM = 'remote_ra_qwen_goal_rewriter'


def upstream():
    """The vendored upstream rewriter script (dram_kv_scripts/qwen_goal_rewriter.py) as a module."""
    module = sys.modules.get(_UPSTREAM)
    if module is None:
        path = Path(__file__).resolve().parents[1] / 'dram_kv_scripts' / 'qwen_goal_rewriter.py'
        spec = importlib.util.spec_from_file_location(_UPSTREAM, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_UPSTREAM] = module
        spec.loader.exec_module(module)
    return module


def log(message):
    sys.stderr.write('goal rewriter batch child: %s\n' % message)
    sys.stderr.flush()


# ---- one request: the upstream generate up to its model call --------------------------------

class _Captured(Exception):
    def __init__(self, arguments):
        super().__init__('model.generate arguments captured')
        self.arguments = arguments


class _Recorder:
    """The model of ``prepare``'s copy: its ``generate`` records the call instead of running it."""

    def generate(self, **arguments):
        raise _Captured(arguments)


def prepare(generator, request):
    """The arguments the upstream ``QwenGoalGenerator.generate`` passes to ``model.generate``
    for ``request`` (its prompt tensors, greedy settings and its own prefix function); an
    error the upstream code raises before the model call propagates."""
    shim = copy.copy(generator)
    shim.model = _Recorder()
    try:
        upstream().QwenGoalGenerator.generate(shim, request)
    except _Captured as captured:
        return captured.arguments
    raise RuntimeError('the upstream generate returned without calling the model')


def alone(generator, request):
    """The upstream generate of one request: its result object, or the exception it raised."""
    try:
        return upstream().QwenGoalGenerator.generate(generator, request)
    except Exception as error:      # the upstream serve loop answers it as an error line
        return error


# ---- several requests: one model.generate ----------------------------------------------------

ARGUMENTS = frozenset({'input_ids', 'attention_mask', 'do_sample', 'max_new_tokens', 'pad_token_id',
                       'prefix_allowed_tokens_fn'})
SHARED = ('do_sample', 'max_new_tokens', 'pad_token_id')


def stop_ids(generator):
    """The eos ids at which Transformers' generate stops a row (the model's generation_config)."""
    value = getattr(getattr(generator.model, 'generation_config', None), 'eos_token_id', None)
    values = value if isinstance(value, (list, tuple)) else () if value is None else (value,)
    return frozenset(int(v) for v in values)


class _Row:
    __slots__ = ('function', 'offset', 'length', 'done')

    def __init__(self, function, offset, length):
        self.function, self.offset, self.length, self.done = function, offset, length, False


def finish(generator, suffix, input_tokens, generation_latency_ms):
    """The upstream generate's end for one row: decode, telemetry, parse."""
    U = upstream()
    response_text = generator.tokenizer.decode(suffix, skip_special_tokens=True)
    telemetry = {
        "input_tokens": int(input_tokens),
        "output_tokens": int(suffix.shape[0]),
        "generation_latency_ms": generation_latency_ms,
    }
    try:
        candidate = U.parse_exact_candidate(response_text)
    except Exception as error:
        failed = U.GenerationFailed(f"{type(error).__name__}: {error}", telemetry)
        failed.__cause__ = error
        return failed
    return {"candidate": candidate, "model_cost": 0.0, **telemetry}


def generate_batch(generator, prepared):
    """One ``model.generate`` for the ``prepare``d requests: their result objects or
    GenerationFailed, in order. Raises what the model call raises."""
    torch = generator.torch
    first = prepared[0]
    for arguments in prepared:
        if (set(arguments) != ARGUMENTS or arguments['input_ids'].shape[0] != 1 or
                any(arguments[k] != first[k] for k in SHARED)):
            raise RuntimeError('the upstream generate arguments differ between the batched requests')
    pad = first['pad_token_id']
    lengths = [int(arguments['input_ids'].shape[1]) for arguments in prepared]
    width = max(lengths)
    ids = torch.full((len(prepared), width), pad, dtype=first['input_ids'].dtype, device=first['input_ids'].device)
    mask = torch.zeros((len(prepared), width), dtype=first['attention_mask'].dtype,
                       device=first['attention_mask'].device)
    rows = []
    for k, (arguments, length) in enumerate(zip(prepared, lengths)):
        ids[k, width - length:] = arguments['input_ids'][0]
        mask[k, width - length:] = arguments['attention_mask'][0]
        rows.append(_Row(arguments['prefix_allowed_tokens_fn'], width - length, length))
    stops = stop_ids(generator)

    def allowed(batch_id, sent):
        row = rows[batch_id]
        if row.done:
            return [pad]
        own = sent[row.offset:]
        if own.shape[0] > row.length and int(own[-1]) in stops:
            row.done = True             # finished: Transformers pads this row from now on
            return [pad]
        return row.function(0, own)     # as alone: batch id 0, the unpadded sequence
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = perf_counter()
    with torch.inference_mode():
        generated = generator.model.generate(input_ids=ids, attention_mask=mask, do_sample=first['do_sample'],
                                             max_new_tokens=first['max_new_tokens'], pad_token_id=pad,
                                             prefix_allowed_tokens_fn=allowed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generation_latency_ms = (perf_counter() - started) * 1_000
    out = []
    for k, (length, tokens) in enumerate(zip(lengths, generated[:, width:].tolist())):
        # What Transformers returns for this request alone: up to its first stop token.
        cut = next((p + 1 for p, token in enumerate(tokens) if token in stops), len(tokens))
        out.append(finish(generator, generated[k, width:width + cut], length, generation_latency_ms))
    return out


def out_of_memory(generator, error):
    kind = getattr(getattr(generator.torch, 'cuda', None), 'OutOfMemoryError', None)
    return (isinstance(kind, type) and isinstance(error, kind)) or 'out of memory' in str(error).lower()


def run(generator, work):
    """``work``: (request, prepared) pairs. Their results or exceptions, in order."""
    if len(work) == 1:
        return [alone(generator, work[0][0])]
    try:
        return generate_batch(generator, [prepared for _, prepared in work])
    except Exception as error:
        if out_of_memory(generator, error):
            cuda = generator.torch.cuda
            if cuda.is_available():
                cuda.empty_cache()
            half = len(work) // 2
            log('out of memory in a batch of %d; running it as %d + %d' % (len(work), half, len(work) - half))
            return run(generator, work[:half]) + run(generator, work[half:])
        log('a batch of %d failed (%s: %s); running its requests alone' % (len(work), type(error).__name__, error))
        return [alone(generator, request) for request, _ in work]


# ---- the protocol ------------------------------------------------------------------------------

def error_answer(request_id, error):
    """The upstream serve loop's error object."""
    response = {
        "request_id": request_id,
        "error": f"{type(error).__name__}: {error}"[:1_000],
    }
    telemetry = getattr(error, "telemetry", None)
    if isinstance(telemetry, Mapping):
        response["telemetry"] = dict(telemetry)
    return response


def answer(generator, envelopes):
    """The upstream answer object of each envelope, in order (the upstream serve loop's
    envelope checks; an error before the model call answers that envelope alone)."""
    answers, work, where = [None] * len(envelopes), [], []
    single = len(envelopes) == 1
    for k, envelope in enumerate(envelopes):
        request_id = ""
        try:
            if not isinstance(envelope, dict) or set(envelope) != {"request_id", "request"}:
                raise ValueError("request envelope has unknown or missing fields")
            request_id = envelope["request_id"]
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("request_id must be a non-empty string")
            request = envelope["request"]
            work.append((request, None if single else prepare(generator, request)))
            where.append((k, request_id))
        except Exception as error:
            answers[k] = error_answer(request_id, error)
    if work:
        for (k, request_id), result in zip(where, run(generator, work)):
            answers[k] = (error_answer(request_id, result) if isinstance(result, BaseException)
                          else {"request_id": request_id, "result": result})
    return answers


def serve(generator, limit, lines=None, write=None):
    """The upstream readiness record, then one answer line per request line."""
    U = upstream()
    lines = sys.stdin if lines is None else lines
    write = U._write_json if write is None else write
    write({"protocol": U.PROTOCOL, "version": U.PROTOCOL_VERSION, "ready": True})
    for line in lines:
        try:
            value = json.loads(line)
        except Exception as error:
            write(error_answer('', error))
            continue
        if isinstance(value, dict) and set(value) == {BATCH}:
            envelopes = value[BATCH]
            if not (isinstance(envelopes, list) and 1 <= len(envelopes) <= limit):
                write({"error": "a batch line holds 1-%d envelopes" % limit})
                continue
            write({BATCH: answer(generator, envelopes)})
        else:
            write(answer(generator, [value])[0])


# ---- startup -----------------------------------------------------------------------------------

def cache_tokenizer_data(generator, probe='{"description":"Retrieve evidence","files":["a.py"]}'):
    """Build lm-format-enforcer's tokenizer data once and let the upstream generate's
    prefix-function build use it. Startup only; on any failure the upstream per-request
    build stays. Returns whether the cache is in use."""
    try:
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        U = upstream()
        build, tokenizer = generator.build_prefix_fn, generator.tokenizer
        data = build_token_enforcer_tokenizer_data(tokenizer)
        schema = U.candidate_schema(constraints_allowed=False)
        fresh = build(tokenizer, generator.JsonSchemaParser(schema))
        cached = build(data, generator.JsonSchemaParser(schema))
        prompt = tokenizer(tokenizer.apply_chat_template(
            [{"role": "user", "content": "probe"}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False), return_tensors="pt")["input_ids"][0]
        sequence = list(map(int, prompt))
        for token in [None] + tokenizer(probe, add_special_tokens=False)["input_ids"]:
            if token is not None:
                sequence.append(int(token))
            sent = generator.torch.tensor(sequence)
            if sorted(map(int, fresh(0, sent))) != sorted(map(int, cached(0, sent))):
                raise RuntimeError('cached tokenizer data allows other tokens than a fresh build')
    except Exception as error:
        log('tokenizer data not cached, the upstream per-request build stays (%s: %s)' % (type(error).__name__, error))
        return False

    def build_prefix_fn(tokenizer_, parser):
        if tokenizer_ is not tokenizer:
            return build(tokenizer_, parser)
        return build(data, parser)
    generator.build_prefix_fn = build_prefix_fn
    log('tokenizer data cached (startup self-check passed)')
    return True


def limit_from(argv):
    """``--batch N`` taken out of ``argv`` (the rest are the upstream script's arguments)."""
    if '--batch' not in argv:
        raise SystemExit('--batch N is required')
    k = argv.index('--batch')
    value = argv[k + 1] if k + 1 < len(argv) else ''
    if not value.isdigit() or not 2 <= int(value) <= LIMIT:
        raise SystemExit('--batch N must be an integer 2-%d' % LIMIT)
    del argv[k:k + 2]
    return int(value)


def main():
    U = upstream()
    limit = limit_from(sys.argv)
    args = U.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if not args.serve:
        raise SystemExit('the batched child serves the NDJSON protocol only (--serve)')
    generator = U.QwenGoalGenerator(args)
    cache_tokenizer_data(generator)
    serve(generator, limit)


if __name__ == '__main__':
    main()
