"""Offline CPU check of the stock keep-reasoning history (2026-11-16): recorded SWE-bench conversations rendered the way
a stock vLLM 0.24.0 server renders them (postrun/stock_history.py: vLLM's parse + the pinned Qwen3.6 template + the
tokenizer), with and without the earlier reasoning.

  stock   <pool dir> : a stock run's trajectories (e.g. stock32t of job 3583822). Calibration: every call's recorded
            usage.prompt_tokens must equal the render the run was served (reasoning dropped: no `reasoning` field, no
            preserve_thinking); then the same calls rendered as the keep plans would send and serve them (the
            `reasoning` copy + preserve_thinking true): their lengths, whether each prompt extends the previous one
            token for token (append-only), and the per-call growth prompt[k+1] - prompt[k] - completion[k] of both.
  runtime <pool dir> <job-output/label dir> : our runtime's run (e.g. full262tr of job 3583081, thinking on, keep): every
            call's keep render against the server's own prompt_token_ids (runtime.jsonl), which keep each earlier
            reply's generated ids; a difference is reported with its first token position, both sides decoded around
            it, and whether the two prompts decode to the same text (a tokenization difference) or not.

Run under a Python with transformers (tbench: /home/ubuntu/swe32-verify/venv/tok-*/bin/python):
  python -I -B stock_keep_render.py stock|runtime <args> --tokenizer DIR --template FILE --out FILE [--limit N]
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stock_history as sh  # noqa: E402

TEMPLATE_SHA256 = 'e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259'


def trajectories(pool, limit=None):
    names = (Path(pool) / 'instances.txt').read_text().split()
    for name in names[:limit]:
        path = Path(pool) / 'runs' / name / name / (name + '.traj.json')
        yield name, (sh.load_json(path) if path.is_file() else None)


def summary(values):
    if not values:
        return None
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return dict(n=len(values), min=values[0], p05=pick(0.05), median=statistics.median(values), p95=pick(0.95),
                max=values[-1], negative=sum(1 for v in values if v < 0))


def check_stock(args, tok, template):
    out = dict(mode='stock', pool=str(args.pool), instances={}, calls=0, dropped_equal=0, kept_equal_recorded=0,
               differ_dropped=[], kept_longer=0, kept_append_breaks=0, dropped_append_breaks=0)
    growth = dict(recorded_dropped=[], kept=[], recorded_dropped_after_reply=[], kept_after_reply=[])
    added = []
    for name, traj in trajectories(args.pool, args.limit):
        if traj is None:
            out['instances'][name] = 'no trajectory'
            continue
        messages = traj['messages']
        calls = sh.agent_calls(traj)
        prev = None
        item = dict(calls=len(calls), dropped_equal=0, kept_tokens=0, dropped_tokens=0)
        for k, call in enumerate(calls):
            history = messages[:call['index']]
            dropped = sh.render_ids(tok, template, history, send_reasoning=False, server_default=None)
            kept = sh.render_ids(tok, template, history, send_reasoning=True, server_default=sh.KEEP_KWARGS)
            recorded = call['usage'].get('prompt_tokens')
            completion = call['usage'].get('completion_tokens')
            out['calls'] += 1
            if recorded == len(dropped):
                out['dropped_equal'] += 1; item['dropped_equal'] += 1
            elif len(out['differ_dropped']) < 20:
                out['differ_dropped'].append(dict(instance=name, call=k, recorded=recorded, dropped=len(dropped),
                                                  kept=len(kept)))
            if recorded == len(kept):
                out['kept_equal_recorded'] += 1
            out['kept_longer'] += len(kept) > len(dropped)
            added.append(len(kept) - len(dropped))
            item['kept_tokens'] += len(kept); item['dropped_tokens'] += len(dropped)
            if prev is not None:
                if kept[:len(prev['kept'])] != prev['kept']:
                    out['kept_append_breaks'] += 1
                if dropped[:len(prev['dropped'])] != prev['dropped']:
                    out['dropped_append_breaks'] += 1
                if isinstance(prev['completion'], int):
                    after = '_after_reply' if prev['kind'] == 'assistant' else ''
                    if isinstance(recorded, int) and isinstance(prev['recorded'], int):
                        growth['recorded_dropped'].append(recorded - prev['recorded'] - prev['completion'])
                        if after:
                            growth['recorded_dropped' + after].append(recorded - prev['recorded'] - prev['completion'])
                    growth['kept'].append(len(kept) - len(prev['kept']) - prev['completion'])
                    if after:
                        growth['kept' + after].append(len(kept) - len(prev['kept']) - prev['completion'])
            prev = dict(kept=kept, dropped=dropped, recorded=recorded, completion=completion, kind=call['kind'])
        out['instances'][name] = item
    out['growth'] = {k: summary(v) for k, v in growth.items()}
    out['kept_minus_dropped_tokens'] = summary(added)
    out['status'] = 'CALIBRATION_PASS' if out['calls'] and out['dropped_equal'] == out['calls'] else 'CALIBRATION_FAIL'
    return out


def context(tok, ids, at, width=8):
    return dict(tokens=ids[max(0, at - width):at + width], text=tok.decode(ids[max(0, at - width):at + width]))


def check_runtime(args, tok, template):
    rows = {}
    with open(Path(args.evidence) / 'runtime.jsonl', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            rows.setdefault(row['session_index'], []).append(row)
    # A call "after a rebuild": its session had a rebuild row (thinking.history 'rebuild' or stage_b.rebuild) at or
    # before it. The XML protocol's rebuild re-encodes the earlier replies from their content (no reasoning), and the
    # appends after it extend that prompt, so such prompts are not the keep history (docs/running.md 6.8).
    out = dict(mode='runtime', pool=str(args.pool), evidence=str(args.evidence), calls=0, matched=0, exact=0,
               same_text=0, differ=[], differ_before_rebuild=[], instances={}, kinds={},
               before_rebuild=dict(calls=0, exact=0, same_text=0, prompt_tokens=0),
               after_rebuild=dict(calls=0, exact=0, same_text=0, prompt_tokens=0), sessions_with_rebuild=0)
    for session, (name, traj) in enumerate(trajectories(args.pool, args.limit)):
        if traj is None:
            out['instances'][name] = 'no trajectory'
            continue
        by_id = {r['response_id']: r for r in rows.get(session, [])}
        messages = traj['messages']
        item = dict(calls=0, exact=0, same_text=0, first_rebuild_call=None)
        rebuilt = False
        for k, call in enumerate(sh.agent_calls(traj)):
            row = by_id.get(call['response_id'])
            out['calls'] += 1; item['calls'] += 1
            if row is None:
                continue
            out['matched'] += 1
            kept = sh.render_ids(tok, template, messages[:call['index']], send_reasoning=True,
                                 server_default=sh.KEEP_KWARGS)
            server = row['prompt_token_ids']
            history = (row.get('thinking') or {}).get('history')
            if not rebuilt and (history == 'rebuild' or (row.get('stage_b') or {}).get('rebuild')):
                rebuilt = True
                item['first_rebuild_call'] = k
                out['sessions_with_rebuild'] += 1
            phase = out['after_rebuild' if rebuilt else 'before_rebuild']
            phase['calls'] += 1; phase['prompt_tokens'] += len(server)
            kind = out['kinds'].setdefault(str(history), dict(calls=0, exact=0, same_text=0))
            kind['calls'] += 1
            if kept == server:
                out['exact'] += 1; item['exact'] += 1; kind['exact'] += 1; phase['exact'] += 1
                continue
            text_kept, text_server = tok.decode(kept), tok.decode(server)
            same = text_kept == text_server
            out['same_text'] += same; item['same_text'] += same; kind['same_text'] += same; phase['same_text'] += same
            target = out['differ'] if rebuilt else out['differ_before_rebuild']
            if len(target) < args.examples:
                first = next((i for i, (a, b) in enumerate(zip(kept, server)) if a != b), min(len(kept), len(server)))
                char = next((i for i, (a, b) in enumerate(zip(text_kept, text_server)) if a != b),
                            min(len(text_kept), len(text_server)))
                target.append(dict(
                    instance=name, call=k, kind=call['kind'], history=history, kept_tokens=len(kept),
                    server_tokens=len(server), first_token_difference=first, same_text=same,
                    kept_context=context(tok, kept, first), server_context=context(tok, server, first),
                    first_char_difference=None if same else char,
                    kept_chars=None if same else repr(text_kept[max(0, char - 40):char + 40]),
                    server_chars=None if same else repr(text_server[max(0, char - 40):char + 40])))
        out['instances'][name] = item
    before = out['before_rebuild']
    out['status'] = ('TOKEN_EXACT' if out['matched'] and out['exact'] == out['matched'] else
                     'TOKEN_EXACT_BEFORE_REBUILD' if before['calls'] and before['exact'] == before['calls'] else
                     'DIFFERENCES')
    return out


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('stock', 'runtime'))
    parser.add_argument('pool')
    parser.add_argument('evidence', nargs='?')
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--template', help='the pinned template file (default: the template of the tokenizer '
                        'config, which must be the pinned one)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--examples', type=int, default=40)
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    template = Path(args.template).read_bytes().decode('utf-8') if args.template else sh.pinned_template(args.tokenizer)
    if hashlib.sha256(template.encode('utf-8')).hexdigest() != TEMPLATE_SHA256:
        raise SystemExit('template is not the pinned Qwen3.6 template e84f32a2')
    value = check_stock(args, tok, template) if args.mode == 'stock' else check_runtime(args, tok, template)
    import transformers
    value.update(transformers=transformers.__version__, template_sha256=TEMPLATE_SHA256)
    Path(args.out).write_text(json.dumps(value, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in value.items() if k not in ('instances', 'differ', 'differ_dropped',
                                                                  'differ_before_rebuild')}, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
