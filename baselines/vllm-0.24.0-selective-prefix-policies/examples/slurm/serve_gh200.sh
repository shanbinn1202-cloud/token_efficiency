#!/usr/bin/env bash
# One selective-prefix server on one GH200 in a Slurm job: environment, serve.py start, READY, a harness, SIGTERM.
# Modelled on the validated jobs (clean-install acceptance 3270721; SWE-bench pilots 3299922 and 3300434 at 784-token
# pages, 3313802, 3314010, 3314319 and 3314623 at 16-token pages), which passed these options on the sbatch command
# line (see README, "GPU job"):
#   sbatch --account=<account> --partition=<gh200 partition> --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16 \
#     --mem=200g --time=<hh:mm:ss> --job-name=<name> --output=<run root>/slurm-%j.out --export=ALL \
#     --mail-type=BEGIN,END,FAIL --mail-user=<email> examples/slurm/serve_gh200.sh
#
# Settings come from the submitting shell (--export=ALL):
#   REPO        this repository (absolute)                  VENV        the base vLLM environment (venv prefix)
#   RELEASE     the release made by install.py --apply      TASKS       task JSON, or a list of 1-16 tasks
#   POLICY      full | random50 | goal_query | observation_query | random_fill
#   RUN_ROOT    short existing directory for this job's files (TMPDIR lives under it: keep it short)
#   KV_GIB      8 (default), 16, 24 or 32                   PORT        server port (default 8000)
#   REWRITER    Qwen3-0.6B directory (goal_query; or set NO_GOAL_REWRITER=1)
#   MAX_RUNNING session pool (serve.py --max-running-sessions): TASKS lists 1-64 tasks, at most MAX_RUNNING (1-16)
#               sessions run at once; MODE=swebench passes it to run_agents.sh, whose K agents refill from the list
#   PAGE_TOKENS 16, 32, 64 or 784 (serve.py --page-tokens)  KV_LAYOUT   decoupled (serve.py --kv-layout)
#               both unset: the default 784-token layout; recommended for goal_query, observation_query and
#               random_fill: PAGE_TOKENS=16 KV_LAYOUT=decoupled (README section 4.6)
#   MAX_TOKENS  1536 or 4096 (serve.py --max-tokens; MODE=swebench also passes it to run_agents.sh)
#   MAX_MODEL_LEN 65536, 131072 or 262144 (serve.py --max-model-len)   DRAM_ARENA_GIB  serve.py --dram-arena-gib
#   STEP_LIMIT  MODE=swebench: the agents' step limit (run_agents.sh; default 90)
#               all unset: the server and the agents run exactly as before (not GPU-validated otherwise)
#   MODE        smoke (default: one examples/http_client.py request; TASKS must be examples/task.json)
#               swebench (reverse tunnel to the agent host, then run_agents.sh there over SSH; see below)
# MODE=swebench also needs: AGENT_HOST (user@host), TUNNEL_KEY (private key path), KNOWN_HOSTS (file pinning the
# agent host's key), AGENT_SOCKET (new socket path in a 0700 directory on the agent host), AGENT_PORT (loopback port
# for relay.py there), AGENT_RUN (path of examples/swebench/run_agents.sh there), AGENT_OUT (new output directory
# there), AGENT_SECONDS (time limit of the agent run) and, on the agent host, the run_agents.sh environment
# (MINI_PYTHON, MSWEA_CONFIG_DIR, optionally UPSTREAM_DIR) set in AGENT_ENV, e.g. AGENT_ENV='MINI_PYTHON=... MSWEA_CONFIG_DIR=...'.
set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}" || { echo "run this under sbatch" >&2; exit 2; }
: "${REPO:?}" "${VENV:?}" "${RELEASE:?}" "${TASKS:?}" "${POLICY:?}" "${RUN_ROOT:?}"
KV_GIB=${KV_GIB:-8}; PORT=${PORT:-8000}; MODE=${MODE:-smoke}; READY_SECONDS=${READY_SECONDS:-900}

# Job-local directories; vLLM binds ZMQ IPC sockets at $TMPDIR/<36-byte uuid> (AF_UNIX limit 107 bytes).
STAGE="$RUN_ROOT/$SLURM_JOB_ID"
mkdir "$STAGE" "$STAGE/tmp" "$STAGE/home" "$STAGE/cache"
(( ${#STAGE} + 4 + 37 <= 107 )) || { echo "TMPDIR too long for AF_UNIX: $STAGE/tmp" >&2; exit 2; }
export TMPDIR="$STAGE/tmp" TMP="$STAGE/tmp" TEMP="$STAGE/tmp"
export HOME="$STAGE/home" XDG_CACHE_HOME="$STAGE/cache"
export HF_HOME="$STAGE/cache/hf" TORCH_HOME="$STAGE/cache/torch"
export TRITON_CACHE_DIR="$STAGE/cache/triton" CUDA_CACHE_PATH="$STAGE/cache/cuda"
export TORCH_EXTENSIONS_DIR="$STAGE/cache/torch-extensions"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Native top-k/top-p sampler: no FlashInfer sampler JIT against the empty job-local cache.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Triton uses $CC before gcc/clang; some login shells export CC=cc.
unset CC VLLM_RPC_BASE_PATH
# HOME is job-local, so ssh gets the key and the pinned host key by absolute path, never from an agent.
unset SSH_AUTH_SOCK SSH_AGENT_PID
export PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_PRELOAD
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
PY="$VENV/bin/python"

SELECTOR=()
if [[ $POLICY == goal_query ]]; then
  if [[ ${NO_GOAL_REWRITER:-0} == 1 ]]; then SELECTOR=(--no-goal-rewriter); else SELECTOR=(--goal-rewriter-model "${REWRITER:?set REWRITER or NO_GOAL_REWRITER=1}"); fi
fi
# Optional KV layout; with PAGE_TOKENS and KV_LAYOUT unset, serve.py gets no layout flag (default 784-token layout).
LAYOUT=()
if [[ -n ${PAGE_TOKENS:-} ]]; then LAYOUT+=(--page-tokens "$PAGE_TOKENS"); fi
if [[ -n ${KV_LAYOUT:-} ]]; then LAYOUT+=(--kv-layout "$KV_LAYOUT"); fi
POOL=()
if [[ -n ${MAX_RUNNING:-} ]]; then POOL=(--max-running-sessions "$MAX_RUNNING"); fi
# Optional context settings; unset, serve.py gets none of these flags.
CONTEXT=()
if [[ -n ${MAX_TOKENS:-} ]]; then CONTEXT+=(--max-tokens "$MAX_TOKENS"); fi
if [[ -n ${MAX_MODEL_LEN:-} ]]; then CONTEXT+=(--max-model-len "$MAX_MODEL_LEN"); fi
if [[ -n ${DRAM_ARENA_GIB:-} ]]; then CONTEXT+=(--dram-arena-gib "$DRAM_ARENA_GIB"); fi
OUTPUT="$STAGE/$POLICY"
SERVE=("$PY" -B "$REPO/serve.py" --release-dir "$RELEASE" --task "$TASKS" --policy "$POLICY"
       --output "$OUTPUT" --port "$PORT" --kv-cache-gib "$KV_GIB" ${SELECTOR[@]+"${SELECTOR[@]}"}
       ${LAYOUT[@]+"${LAYOUT[@]}"} ${POOL[@]+"${POOL[@]}"} ${CONTEXT[@]+"${CONTEXT[@]}"})
"${SERVE[@]}" > "$STAGE/dry-run.json"            # dry run: checks the release, prints the command
"${SERVE[@]}" --start > "$STAGE/server.log" 2>&1 &
SERVER=$!                                         # serve.py execs the APIServer: this is the server PID
TUNNEL=""
stop_tunnel() { if [[ -n $TUNNEL ]]; then kill -TERM "$TUNNEL" 2>/dev/null || true; wait "$TUNNEL" 2>/dev/null || true; TUNNEL=""; fi; }
trap stop_tunnel EXIT

deadline=$((SECONDS + READY_SECONDS))
while true; do
  set +e; "$PY" -B "$REPO/examples/slurm/server_checks.py" ready "$OUTPUT" "$POLICY" "$RELEASE" > "$STAGE/ready-check.txt"; rc=$?; set -e
  (( rc == 0 )) && break
  (( rc == 1 )) || { cat "$STAGE/ready-check.txt" >&2; kill -TERM "$SERVER"; wait "$SERVER" || true; exit 1; }
  kill -0 "$SERVER" 2>/dev/null || { echo "server exited before READY (see $STAGE/server.log)" >&2; wait "$SERVER" || true; exit 1; }
  (( SECONDS < deadline )) || { echo "no READY within $READY_SECONDS s" >&2; kill -TERM "$SERVER"; wait "$SERVER" || true; exit 1; }
  sleep 5
done
echo "READY after $SECONDS s"

harness_rc=0
if [[ $MODE == smoke ]]; then
  "$PY" -B "$REPO/examples/http_client.py" --base-url "http://127.0.0.1:$PORT/v1" \
    --messages "$REPO/examples/messages.json" > "$STAGE/smoke.json" || harness_rc=$?
elif [[ $MODE == swebench ]]; then
  : "${AGENT_HOST:?}" "${TUNNEL_KEY:?}" "${KNOWN_HOSTS:?}" "${AGENT_SOCKET:?}" "${AGENT_PORT:?}" "${AGENT_RUN:?}" "${AGENT_OUT:?}" "${AGENT_SECONDS:?}"
  SSH_OPTS=(-i "$TUNNEL_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes
            -o UserKnownHostsFile="$KNOWN_HOSTS" -o GlobalKnownHostsFile=/dev/null -o ConnectTimeout=20)
  # Reverse tunnel: a Unix socket on the agent host -> this node's 127.0.0.1:$PORT (never a TCP listener there).
  ssh -N -R "$AGENT_SOCKET:127.0.0.1:$PORT" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 "${SSH_OPTS[@]}" "$AGENT_HOST" > "$STAGE/tunnel.log" 2>&1 &
  TUNNEL=$!
  up=0
  for _ in $(seq 1 30); do
    kill -0 "$TUNNEL" 2>/dev/null || break
    if ssh -T "${SSH_OPTS[@]}" "$AGENT_HOST" curl -sf --max-time 5 --unix-socket "$AGENT_SOCKET" \
         http://localhost/v1/models 2>/dev/null | grep -q '"selective-prefix"'; then up=1; break; fi
    sleep 2
  done
  if (( up == 1 )); then
    INSTANCES=$("$PY" -B -c 'import json,sys; t=json.load(open(sys.argv[1])); print(" ".join(x["instance_id"] for x in (t if isinstance(t, list) else [t])))' "$TASKS")
    # An SSH drop or timeout does not stop the remote runner: stop it and its containers on the agent host then.
    timeout "$AGENT_SECONDS" ssh -T "${SSH_OPTS[@]}" "$AGENT_HOST" env ${AGENT_ENV:-} \
      ${MAX_RUNNING:+MAX_RUNNING=$MAX_RUNNING} ${MAX_TOKENS:+MAX_TOKENS=$MAX_TOKENS} \
      ${STEP_LIMIT:+STEP_LIMIT=$STEP_LIMIT} "$AGENT_RUN" \
      "$AGENT_SOCKET" "$AGENT_PORT" "$AGENT_OUT" $INSTANCES > "$STAGE/agents.log" 2>&1 || harness_rc=$?
  else
    echo "tunnel not up (see $STAGE/tunnel.log)" >&2; harness_rc=1
  fi
  stop_tunnel
else
  echo "unknown MODE: $MODE" >&2; harness_rc=2
fi

# Stop: SIGTERM to the server PID only (never the process group), then a zero exit and clean receipts.
kill -TERM "$SERVER"
set +e; wait "$SERVER"; server_rc=$?; set -e
echo "server exit $server_rc, harness exit $harness_rc"
(( server_rc == 0 )) || exit 1
"$PY" -B "$REPO/examples/slurm/server_checks.py" cleanup "$OUTPUT"
exit "$harness_rc"
