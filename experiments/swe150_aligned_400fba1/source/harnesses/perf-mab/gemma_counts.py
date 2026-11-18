"""Local only (never delivered): the Gemma 4 token counts of the real MemAgent32 cases (counts only, no text).

usage (the tokenizer venv; the pinned Gemma metadata directory and the local copy of the data):
  python -B gemma_counts.py <gemma metadata dir> <data dir> <out json>
For every case of pins.real_data (verified against the pin, read as JSON only) the runner's own Gemma client
(mab_runner, grammar gemma4) plays the whole read_next loop against a token-level stand-in of the server that answers
every read with the shortest valid reply (one `bash read_next` call, stopped on <|tool_response>) and the question with
a one-token answer: the served stream of the answer turn, token for token as our runtime would hold it (the first
prompt, every tool response wrapper and chunk, the thought channels kept after each `<|turn>model\\n`, the question and
the header). Recorded per case: chunks, first_prompt_tokens, stream_tokens (N of the answer turn), reply_tokens (the
minimal read replies inside that stream), prompt_tokens = stream_tokens - reply_tokens (everything but the replies:
controller.gemma_arena_fit adds the replies at the plan's bound), max_tail_tokens (the largest continuation body:
one chunk with its wrapper), and the per-case Qwen manifest count for reference. build.py copies the file into
pins.json gemma_real_tokens (with the pinned manifest SHA-256); test_real_tokenizer re-derives it.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import mab_runner as R  # noqa: E402
import mab_workload as W  # noqa: E402

SCHEMA = 'pmab-gemma-real-tokens-v1'
READ = '<|tool_call>call:bash{command:<|"|>read_next<|"|>}<tool_call|>'


def stream_counts(kit, case):
    """The counts of one real case (module docstring) through mab_runner.run_case with a token-server stand-in."""
    state = dict(stream=None, replies=0, tails=[])

    def serve(method, url, body=None, timeout=60):
        if url.endswith('/close'):
            return 200, dict(closed=True), 0, 0
        ids = body['input_ids']
        T = list(ids) if state['stream'] is None else state['stream'] + list(ids[1:])
        if state['stream'] is not None:
            state['tails'].append(len(ids))
        answering = body['messages'][-1]['role'] == 'user' and state['stream'] is not None
        if answering:
            state['answer_prompt'] = len(T)
            out = kit.encode('A') + [kit.eos]
        else:
            out = kit.encode(READ) + [kit.encode(R.GEMMA4_TOOL_OPEN)[0]]
            state['replies'] += len(out)
        state['stream'] = T + out
        return 200, dict(text=kit.decode(out), output_ids=out, prompt_tokens=len(T), completion_tokens=len(out),
                         cached_tokens=len(T) - len(ids), finish_reason='stop', effective_max_tokens=4096), 1, 2
    with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as out:
        spec = dict(schema='pmab-runner-spec-v1', mode='kv', label='count', base_url='http://x', out=out,
                    tokenizer=dict(kind='stand-in'), grammar=R.GEMMA4,
                    generation=dict(max_tokens=4096, max_turns=120, temperature=0, top_p=1.0, top_k=-1,
                                    repetition_penalty=1.0, seed=42, enable_thinking=False,
                                    max_consecutive_format_errors=2, request_timeout=3600, max_window_size=262144),
                    budget=None, expect_info={}, workers=1, agent_seconds=3600, drain_seconds=60, close_seconds=10)
        with patch.object(R, 'make_kit', lambda spec, grammar=None: kit), patch.object(R, 'http', serve):
            runner = R.Runner(spec)
            row = R.run_case(runner, case, 0)
            runner.recorder.close()
            first = json.loads((Path(out) / 'client' / 'requests.jsonl').read_text(encoding='utf-8').splitlines()[0])
    if row['status'] != 'answered' or row['chunks_read'] != len(case['chunks']):
        raise RuntimeError('%s: the counting loop ended %s after %d of %d chunks' % (
            case['id'], row['status'], row['chunks_read'], len(case['chunks'])))
    stream = state['answer_prompt']
    return dict(id=case['id'], chunks=len(case['chunks']), first_prompt_tokens=len(first['body']['input_ids']),
                stream_tokens=stream, reply_tokens=state['replies'], prompt_tokens=stream - state['replies'],
                max_tail_tokens=max(state['tails']), context_tokens_qwen=case['context_tokens'])


def counts(kit, data_dir, pin):
    """Every pinned case's counts, in the pin's order, with the manifest SHA-256 they belong to."""
    digest, cases = W.load_real(data_dir, pin=pin)
    rows = [stream_counts(kit, case) for case in cases]
    return dict(schema=SCHEMA, manifest_sha256=digest, tokenizer_template_sha256=kit.identity['template_sha256'],
                reply_text=READ, cases=rows)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print('usage: python -B gemma_counts.py <gemma metadata dir> <data dir> <out json>')
        return 2
    meta, data, out = (Path(a) for a in argv)
    pins = json.loads((ROOT / 'pins.json').read_bytes())
    kit = R.hf_kit(meta, R.GEMMA4)
    value = counts(kit, data, pins['real_data'])
    Path(out).write_text(json.dumps(value, indent=1, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    rows = value['cases']
    print('GEMMA_COUNTS cases=%d prompt_tokens %d..%d stream %d..%d max_tail %d first %s' % (
        len(rows), min(r['prompt_tokens'] for r in rows), max(r['prompt_tokens'] for r in rows),
        min(r['stream_tokens'] for r in rows), max(r['stream_tokens'] for r in rows),
        max(r['max_tail_tokens'] for r in rows), sorted({r['first_prompt_tokens'] for r in rows})))
    return 0


if __name__ == '__main__':
    sys.exit(main())
