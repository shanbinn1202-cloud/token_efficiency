#!/usr/bin/env bash
# The 16-agent stress test of the swe32 tbench half (one-time, no GPU, no DeltaAI): bin/fake_server.py on a Unix
# socket stands in for a server at the far end of the tunnel; bin/run_pool.sh runs the real pool (relay, `ss` check,
# K agents with Docker containers, refill, session close, timeline, host sampler, tbench lock) over the first N tasks
# with a short step limit. Prints a JSON summary and the lines to send back: host peaks (CPU, memory, containers,
# disk), per-instance wall, failures, containers left, and what litellm sent (body keys, retry/timeout headers).
# Usage: stress_test.sh [tasks=20] [K=16] [step limit=30] [min delay s=2] [max delay s=8]
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
N=${1:-20}; K=${2:-16}; STEPS=${3:-30}; DMIN=${4:-2}; DMAX=${5:-8}
[[ $N =~ ^[0-9]+$ && $K =~ ^[0-9]+$ && $STEPS =~ ^[0-9]+$ ]] || { echo "usage: $0 [tasks] [K] [step limit]" >&2; exit 2; }
RUN=stress-$(date -u +%Y%m%dT%H%M%S); LABEL=stress; PORT=${SWE32_STRESS_PORT:-22990}
mapfile -t ALL < <(hpy -c 'import json,sys; [print(i) for i in json.load(open(sys.argv[1]))["instances"]]' "$SWE32_ROOT/tasks/swe32-tasks.json")
INSTANCES=("${ALL[@]:0:$N}")
mkdir -p "$RUNS/$RUN"
if [[ ${SWE32_ALLOW_TCP:-0} == 1 ]]; then SOCK=-; TARGET=tcp:$PORT; else SOCK=$SOCKS/$RUN-$LABEL.sock; TARGET=$SOCK; fi
hpy "$SWE32_BIN/fake_server.py" "$TARGET" "$RUNS/$RUN/fake-requests.jsonl" "$DMIN" "$DMAX" \
  $(( STEPS > 12 ? STEPS - 12 : 2 )) $(( STEPS > 4 ? STEPS - 4 : STEPS )) > "$RUNS/$RUN/fake-server.log" 2>&1 &
FAKE=$!
trap 'kill $FAKE 2>/dev/null; wait $FAKE 2>/dev/null; [[ $SOCK != - ]] && rm -f "$SOCK"' EXIT
sleep 2
SPEC=$(hpy - "$SWE32_ROOT" "$PORT" "$K" "$STEPS" "${RUNNER_FILE:-swebench_runner.py}" "${PROMPT_FILE:-swebench_prompt_override.yaml}" <<'PY'
import base64, hashlib, json, sys, time
from pathlib import Path
root, port, k, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
runner_file, prompt_file = sys.argv[5], sys.argv[6]
up = Path(root) / 'upstream'
override = '\n'.join(['model:', '  model_name: "openai/selective-prefix"', '  model_class: litellm_textbased',
    '  cost_tracking: ignore_errors', '  model_kwargs:', '    api_base: "http://127.0.0.1:%d/v1"' % port,
    '    api_key: "dummy"', '    temperature: 0.0', '    max_tokens: 4096', '    timeout: 3600', '    max_retries: 0',
    '    num_retries: 0', '', 'environment:', '  environment_class: docker', '  cwd: "/testbed"', '  timeout: 120',
    '  interpreter: ["bash", "-c"]', '  env:', '    PAGER: cat', '    MANPAGER: cat', '    LESS: -R',
    '    PIP_PROGRESS_BAR: "off"', '    TQDM_DISABLE: "1"', '', 'agent:', '  step_limit: %d' % steps, '  cost_limit: 0', ''])
spec = dict(schema='swe32-pool-spec-v1', label='stress', kind='runtime', max_running=k, step_limit=steps,
            deadline_epoch=int(time.time()) + 3 * 3600,
            runner_sha256=hashlib.sha256((up / runner_file).read_bytes()).hexdigest(),
            prompt_file=prompt_file, prompt_sha256=hashlib.sha256((up / prompt_file).read_bytes()).hexdigest(),
            mini_retry_attempts=1, request_timeout=3600, close=True, model_override=override)
print(base64.urlsafe_b64encode(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).decode().rstrip('='))
PY
)
started=$(date +%s)
"$SWE32_BIN/run_pool.sh" "$RUN" "$LABEL" runtime "$PORT" "$SOCK" "$SPEC" "${INSTANCES[@]}" > "$RUNS/$RUN/pool.stdout" 2> "$RUNS/$RUN/pool.stderr"
rc=$?
hpy - "$RUNS/$RUN" "$rc" "$(( $(date +%s) - started ))" <<'PY'
import json, sys
from pathlib import Path
run, rc, seconds = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
final = None
for line in (run / 'pool.stdout').read_text(errors='replace').splitlines():
    if line.startswith('{'):
        try:
            final = json.loads(line)
        except ValueError:
            pass
requests = [json.loads(l) for l in (run / 'fake-requests.jsonl').read_text().splitlines() if l.strip()] \
    if (run / 'fake-requests.jsonl').is_file() else []
samples = [json.loads(l) for l in (run / 'stress' / 'host-samples.jsonl').read_text().splitlines() if l.strip()] \
    if (run / 'stress' / 'host-samples.jsonl').is_file() else []
try:
    host = json.loads((run / 'stress' / 'host-check.json').read_text())
except (OSError, ValueError):
    host = {}
inst = (final or {}).get('instances') or {}
walls = sorted(v.get('seconds') or 0 for v in inst.values())
headers = {}
for r in requests:
    for k, v in (r.get('headers') or {}).items():
        if k.startswith('x-stainless-retry') or k.startswith('x-stainless-read-timeout') or k.startswith('x-stainless-timeout'):
            headers.setdefault(k, set()).add(v)
peak = lambda key: max([s[key] for s in samples if isinstance(s.get(key), (int, float))], default=None)
summary = dict(event='swe32_stress', pool_exit=rc, seconds=seconds, run=str(run), instances=len(inst),
               done=sum(1 for v in inst.values() if v.get('trajectory')),
               exit_statuses=sorted({str(v.get('exit_status')) for v in inst.values()}),
               close_http=sorted({v.get('close_http') for v in inst.values()}, key=str),
               wall_s=dict(min=walls[0] if walls else None, median=walls[len(walls) // 2] if walls else None,
                           max=walls[-1] if walls else None),
               span_s=(final or {}).get('span_s'), containers_left=(final or {}).get('containers_left'),
               requests=len(requests), max_request_bytes=max([r['bytes'] for r in requests], default=None),
               request_keys=sorted({k for r in requests for k in r.get('keys') or ()}),
               sampling_fields=sorted({k for r in requests for k in (r.get('sampling') or {})}),
               client_headers={k: sorted(v) for k, v in headers.items()},
               cpu_busy_percent_max=peak('cpu_busy_percent'), mem_used_gb_max=peak('mem_used_gb'),
               mem_total_gb=peak('mem_total_gb'), containers_max=peak('containers'), load1_max=peak('load1'),
               disk_free_gb_min=min([s['disk_free_gb'] for s in samples if isinstance(s.get('disk_free_gb'), (int, float))],
                                    default=None), host_samples=len(samples),
               host_check={k: host.get(k) for k in ('status', 'reasons', 'other_containers', 'our_leftovers', 'load',
                                                    'mem_available_gb', 'disk_free_gb')})
summary['verdict'] = 'STRESS_PASS' if (rc == 0 and summary['done'] == summary['instances'] and summary['containers_left'] == 0
                                       and (summary['mem_used_gb_max'] or 0) < 24) else 'STRESS_CHECK'
print(json.dumps(summary, sort_keys=True))
print('SEND BACK: verdict=%s done=%s/%s pool_exit=%s span_s=%s mem_used_gb_max=%s cpu_max=%s containers_max=%s '
      'containers_left=%s disk_free_gb_min=%s max_request_bytes=%s client_headers=%s host_check=%s other_containers=%s'
      % (summary['verdict'], summary['done'], summary['instances'], rc, summary['span_s'], summary['mem_used_gb_max'],
         summary['cpu_busy_percent_max'], summary['containers_max'], summary['containers_left'],
         summary['disk_free_gb_min'], summary['max_request_bytes'], summary['client_headers'],
         host.get('status'), host.get('other_containers')))
PY
exit $rc
