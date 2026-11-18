"""G4 predictions in the MemoryAgentBench scorers' per-case format (offline, from a verified job's evidence).

For every server of the job whose workload is real (pins.config.workloads[...].kind == 'real'), one
`<out>/<label>/cases/<case id>/result.json` per case the runner finished, shaped like the harness's own result rows
(id, split, source, metric, status, error, prediction, context_sha256, context_tokens, chunks_read, total_chunks,
started, finished, wall_seconds, agent_output_tokens, answer_output_tokens, auxiliary_output_tokens, requests,
auxiliary_requests, compression_stats) with `grading_pending: true`, `score: null`, `metrics: null`, `parsed: null`
and `judge_pending` (metric llm_judge_f1): postrun/mab_grade.py applies the official post_process later. Not copied:
the answers, the question and the query (private data; the grader reads them from the manifest). `status` keeps the
harness's vocabulary (answered, answer_length_limit, unfinished_thinking, format_error, turn_limit); every other
runner status (server_error, context_limit, transport_error, stopped_cap, ...) is `error`, the runner's own status
stays in `runner_status`. Cases that never started get no file (not completed). Plus `<out>/<label>/summary.json`.

Usage: python -I -B postrun/mab_predictions.py --job-output <dir with <label>/client/> --pins <pins.json> --out <dir>
"""
import argparse
import json
from pathlib import Path

SCHEMA = 'pmab-prediction-v1'
OFFICIAL_STATUSES = ('answered', 'answer_length_limit', 'unfinished_thinking', 'format_error', 'turn_limit')


def load_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    tmp.replace(path)


def real_servers(pins):
    config = pins['config']
    for server in config['servers']:
        workload = config['workloads'].get(server.get('workload') or '')
        if workload and workload['kind'] == 'real':
            yield server


def request_summary(row):
    response = row.get('response') or {}
    usage = response.get('usage') if isinstance(response.get('usage'), dict) else response
    seconds = None
    if row.get('t_send_ns') is not None and row.get('t_recv_ns') is not None:
        seconds = round((row['t_recv_ns'] - row['t_send_ns']) / 1e9, 3)
    return dict(turn=row.get('turn'), kind=row.get('kind'), status=row.get('status'), seconds=seconds,
                usage=dict(prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'),
                           cached_tokens=usage.get('cached_tokens')),
                finish_reason=response.get('finish_reason'))


def build_server(server, client, out, pins):
    """The result rows of one real-data server from its client records; returns its summary."""
    label = server['label']
    pinned = {c['id']: c for c in pins['real_data']['cases']}
    planned = [c['id'] for c in pins['workload_plan']['workloads'][server['workload']]['cases']]
    runner = json.loads((client / 'runner.json').read_text(encoding='utf-8')) if (client / 'runner.json').is_file() else {}
    requests = {r['seq']: r for r in load_jsonl(client / 'requests.jsonl') if r.get('seq') is not None}
    rows = [r for r in load_jsonl(client / 'cases.jsonl') if r.get('case') in pinned]
    started_ns, started_wall = runner.get('started_ns'), runner.get('started_wall')

    def wall(ns):
        return None if None in (ns, started_ns, started_wall) else round(started_wall + (ns - started_ns) / 1e9, 3)
    written, statuses = [], {}
    for row in rows:
        case = pinned[row['case']]
        if row.get('context_sha256') != case['context_sha256']:
            raise ValueError('%s: context SHA-256 differs from pins.real_data' % row['case'])
        official = row['status'] if row['status'] in OFFICIAL_STATUSES else 'error'
        error = row.get('error')
        if official == 'error' and error is None:
            error = dict(type='RunnerStatus', message=row['status'])
        result = dict(
            schema=SCHEMA, id=row['case'], split=row.get('split') or case['split'], source=row.get('source'),
            metric=row.get('metric'), status=official, runner_status=row['status'], error=error,
            prediction=row.get('prediction') or '', context_sha256=case['context_sha256'],
            context_tokens=case['context_tokens'], chunks_read=row.get('chunks_read'), total_chunks=row.get('total_chunks'),
            started=wall(row.get('started_ns')), finished=wall(row.get('finished_ns')), wall_seconds=row.get('wall_s'),
            agent_output_tokens=row.get('agent_output_tokens'), answer_output_tokens=row.get('answer_output_tokens'),
            auxiliary_output_tokens=0,
            requests=[request_summary(requests[s]) for s in row.get('request_seqs') or () if s in requests],
            auxiliary_requests=[], compression_stats=None,
            method=dict(server=label, kind=server['kind'], policy=server.get('policy'),
                        policy_args=[a for a in server.get('policy_args') or () if not a.startswith('/')],
                        mode=row.get('mode'), workers=server.get('workers')),
            grading_pending=True, score=None, metrics=None, parsed=None,
            judge_pending=row.get('metric') == 'llm_judge_f1')
        write_json(out / label / 'cases' / row['case'] / 'result.json', result)
        written.append(row['case'])
        statuses[official] = statuses.get(official, 0) + 1
    summary = dict(schema=SCHEMA, server=label, planned=len(planned), planned_ids=planned, written=len(written),
                   not_started=[c for c in planned if c not in written], statuses=statuses,
                   runner_outcome=runner.get('outcome'), runner_stop_reason=runner.get('stop_reason'))
    write_json(out / label / 'summary.json', summary)
    return summary


def build(job_output, out, pins):
    """Every real-data server of the job that left client records; returns {label: summary}."""
    job_output, out = Path(job_output), Path(out)
    done = {}
    for server in real_servers(pins):
        client = job_output / server['label'] / 'client'
        if (client / 'cases.jsonl').is_file():
            done[server['label']] = build_server(server, client, out, pins)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--job-output', required=True, type=Path)
    parser.add_argument('--pins', required=True, type=Path, help='the job\'s sealed pins.json (from its archive)')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    pins = json.loads(args.pins.read_text(encoding='utf-8'))
    value = build(args.job_output, args.out, pins)
    print(json.dumps(dict(status='PREDICTIONS_WRITTEN' if value else 'NO_REAL_DATA_SERVER', servers=value), indent=1))


if __name__ == '__main__':
    main()
