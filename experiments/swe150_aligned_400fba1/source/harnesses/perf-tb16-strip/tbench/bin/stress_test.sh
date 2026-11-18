#!/usr/bin/env bash
# The stress test of the tb16 tbench half (one-time per deploy; no GPU, no DeltaAI): real Harbor 0.21.0 and the real,
# unchanged Terminus 2 against bin/fake_server.py on a Unix socket (where the job's reverse tunnel would end), through
# the real bin/run_harbor.sh (spec check, Harbor RECORD and task-package check, the shared locks, host check, relay,
# `ss` check, host sampler, watchdog, Harbor's own job directory, the final report). The job config is the sealed one
# (tbench/stress-config.json, written by build.py from tb16-config.json) with the stress test's port. Prints a JSON
# summary and the line to send back: Harbor's exit, trials done of planned, exceptions, containers left, what Terminus
# 2 / litellm sent (body keys, sampling fields, an X-Session-ID header if any, the first message's role), host peaks.
# Usage: stress_test.sh [tasks=4] [n_concurrent=4] [task name ...]   (default tasks: the first N of the dry subset)
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
N=${1:-4}; K=${2:-4}
shift $(( $# < 2 ? $# : 2 ))
[[ $N =~ ^[0-9]+$ && $K =~ ^[0-9]+$ ]] && (( N >= 1 && K >= 1 && K <= 16 )) || { echo "usage: $0 [tasks] [K] [name ...]" >&2; exit 2; }
NAMES=("$@")
RUN=stress-$(date -u +%Y%m%dT%H%M%S); LABEL=stress; PORT=${TB16_STRESS_PORT:-24990}
mkdir -p "$RUNS/$RUN-fake"
if [[ ${TB16_ALLOW_TCP:-0} == 1 ]]; then SOCK=-; TARGET=tcp:$PORT; else SOCK=$SOCKS/$RUN-$LABEL.sock; TARGET=$SOCK; fi
hpy "$TB16_BIN/fake_server.py" "$TARGET" "$RUNS/$RUN-fake/fake-requests.jsonl" 1 3 2 4 > "$RUNS/$RUN-fake/fake-server.log" 2>&1 &
FAKE=$!
trap 'kill $FAKE 2>/dev/null; wait $FAKE 2>/dev/null; [[ $SOCK != - ]] && rm -f "$SOCK"' EXIT
sleep 2
SPEC=$(hpy - "$TB16_ROOT" "$RUN" "$LABEL" "$PORT" "$N" "$K" "${NAMES[@]}" <<'PY'
import base64, json, sys, time, zlib
from pathlib import Path
root, run, label, port, n, k, *names = sys.argv[1:]
config = json.loads(Path(root, 'stress-config.json').read_text())
tasks = {t['name']: t for t in json.loads(Path(root, 'tasks', 'tb16-tasks.json').read_text())['tasks']}
listing = json.loads(Path(root, 'tasks', 'tb16-tasks.json').read_text())
chosen = names or listing['dry_subset'][:int(n)]
assert all(c in tasks for c in chosen), 'unknown task names'
org = config['dataset']['name'].split('/')[0]
kwargs = dict(config['agent']['kwargs'], api_base='http://127.0.0.1:%s/v1' % port)
job_name = '%s-%s' % (run, label)
job_config = dict(job_name=job_name, jobs_dir=str(Path(root, 'runs', run, label, 'jobs')), n_attempts=1,
                  timeout_multiplier=1.0, n_concurrent_trials=int(k), quiet=True, retry=dict(max_retries=0),
                  environment=dict(type='docker', delete=True, force_build=False), verifier=dict(disable=False),
                  agents=[dict(name=config['agent']['name'], model_name=config['agent']['model_name'], kwargs=kwargs)],
                  datasets=[dict(name=config['dataset']['name'], ref=config['dataset']['ref'],
                                 task_names=['%s/%s' % (org, c) for c in chosen])])
spec = dict(schema='tb16-harbor-spec-v1', run_id=run, label=label, kind='runtime', n_concurrent=int(k),
            deadline_epoch=int(time.time()) + 3 * 3600, job_name=job_name,
            tasks=[dict(name=c, content_hash=tasks[c]['content_hash'], tree_sha256=tasks[c]['tree_sha256'],
                        docker_image=tasks[c]['docker_image'], agent_timeout_sec=tasks[c]['agent_timeout_sec'])
                   for c in chosen], harbor=config['harbor'], session_close='none', telemetry='off',
            job_config=job_config)
raw = json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()
print(base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode().rstrip('='))  # controller.harbor_spec's encoding
PY
) || { echo "stress spec failed" >&2; exit 2; }
started=$(date +%s)
"$TB16_BIN/run_harbor.sh" "$RUN" "$LABEL" runtime "$PORT" "$SOCK" "$SPEC" > "$RUNS/$RUN-fake/run.stdout" 2> "$RUNS/$RUN-fake/run.stderr"
rc=$?
hpy - "$RUNS/$RUN" "$RUNS/$RUN-fake" "$rc" "$(( $(date +%s) - started ))" <<'PY'
import json, sys
from pathlib import Path
run, fake, rc, seconds = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
final = None
for line in (fake / 'run.stdout').read_text(errors='replace').splitlines():
    if line.startswith('{'):
        try:
            final = json.loads(line)
        except ValueError:
            pass
requests = [json.loads(l) for l in (fake / 'fake-requests.jsonl').read_text().splitlines() if l.strip()] \
    if (fake / 'fake-requests.jsonl').is_file() else []
tasks = (final or {}).get('tasks') or {}
summary = dict(event='tb16_stress', run_exit=rc, seconds=seconds, run=str(run), final_event=(final or {}).get('event'),
               harbor_exit=(final or {}).get('harbor_exit'), planned=len(tasks),
               done=sum(1 for v in tasks.values() if v.get('status') == 'done'),
               exceptions=sorted({str(v.get('exception')) for v in tasks.values()}),
               containers_left=(final or {}).get('containers_left'), requests=len(requests),
               asides=sum(1 for r in requests if r.get('aside')),
               session_headers=sorted({r['headers'].get('x-session-id') for r in requests
                                       if (r.get('headers') or {}).get('x-session-id')}),
               first_roles=sorted({(r.get('roles') or ['-'])[0] for r in requests}),
               request_keys=sorted({k for r in requests for k in r.get('keys') or ()}),
               sampling=sorted({json.dumps(r.get('sampling'), sort_keys=True) for r in requests})[:4],
               max_request_bytes=max([r['bytes'] for r in requests], default=None),
               host_peaks=(final or {}).get('host_peaks'), reason=(final or {}).get('reason'))
summary['verdict'] = 'STRESS_PASS' if (rc == 0 and summary['planned'] and summary['done'] == summary['planned']
                                       and summary['containers_left'] == 0 and not summary['session_headers']
                                       and summary['first_roles'] == ['user']) else 'STRESS_CHECK'
print(json.dumps(summary, sort_keys=True))
print('SEND BACK: verdict=%s done=%s/%s harbor_exit=%s exceptions=%s containers_left=%s requests=%s asides=%s '
      'session_headers=%s first_roles=%s request_keys=%s sampling=%s host_peaks=%s'
      % (summary['verdict'], summary['done'], summary['planned'], summary['harbor_exit'], summary['exceptions'],
         summary['containers_left'], summary['requests'], summary['asides'], summary['session_headers'],
         summary['first_roles'], summary['request_keys'], summary['sampling'], summary['host_peaks']))
PY
exit $rc
