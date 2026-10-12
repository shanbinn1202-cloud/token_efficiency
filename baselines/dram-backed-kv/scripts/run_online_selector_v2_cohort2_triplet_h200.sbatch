#!/bin/bash
#SBATCH --job-name=sel-v2-c2
#SBATCH --account=bcjw-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --exclude=gpue06
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=results/raw/online_selector_v2_cohort2_%j.log

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
REPO="${DRAM_KV_REPO_ROOT:-${DEFAULT_REPO}}"
BASE_RUNNER="${REPO}/scripts/run_online_selector_v2_swe_h200.sbatch"
EXPERIMENT_MODE="${EXPERIMENT_MODE:?set EXPERIMENT_MODE to full_context, goal, or observation}"
RUN_GROUP="${RUN_GROUP:?set RUN_GROUP to A or B}"
# Separate allocations can share a physical H200 node. Derive a disjoint pair
# of localhost ports from the job id so their two API servers cannot collide.
base_port=$((20000 + (SLURM_JOB_ID % 10000) * 2))

case "${EXPERIMENT_MODE}" in
  full_context)
    selection_mode="full_context"
    query_source="observation"
    ;;
  goal)
    selection_mode="selector"
    query_source="goal"
    ;;
  observation)
    selection_mode="selector"
    query_source="observation"
    ;;
  *)
    echo "ERROR: unsupported EXPERIMENT_MODE=${EXPERIMENT_MODE}" >&2
    exit 2
    ;;
esac

case "${RUN_GROUP}" in
  A|B) ;;
  *)
    echo "ERROR: unsupported RUN_GROUP=${RUN_GROUP}" >&2
    exit 2
    ;;
esac

run_instance() {
  local worker="$1"
  local port="$2"
  local instance="$3"
  local stem="online_selector_v2_cohort2_${EXPERIMENT_MODE}_${RUN_GROUP}_${SLURM_JOB_ID}_${worker}_${instance}"
  local output_dir="${REPO}/results/raw/${stem}"
  local launcher_log="${REPO}/results/raw/${stem}.log"
  local status=0

  env \
    INSTANCE="${instance}" \
    INSTANCE_FILTER="^${instance}$" \
    PORT="${port}" \
    OUTPUT_DIR="${output_dir}" \
    RUN_SLOT="${worker}" \
    STEP_LIMIT=90 \
    MAX_NEW_TOKENS=1536 \
    SWITCH_THRESHOLD=8192 \
    SELECTION_MODE="${selection_mode}" \
    QUERY_SOURCE="${query_source}" \
    timeout --signal=TERM 5400 \
    bash "${BASE_RUNNER}" >"${launcher_log}" 2>&1 || status=$?
  echo "mode=${EXPERIMENT_MODE} worker=${worker} instance=${instance} status=${status}"

  # Retry only externally terminated children in this allocation. Ordinary
  # runner failures and the explicit 90-minute timeout retain their evidence
  # and advance the queue rather than looping indefinitely.
  if [[ "${status}" -eq 143 || "${status}" -eq 137 ]]; then
    return 75
  fi
  return 0
}

run_supervised() {
  local worker="$1"
  local port="$2"
  local instance="$3"
  while ! run_instance "${worker}" "${port}" "${instance}"; do
    echo "mode=${EXPERIMENT_MODE} worker=${worker} instance=${instance} restarting_in_same_allocation=1"
    sleep 2
  done
}

run_worker_zero() {
  if [[ "${RUN_GROUP}" == "A" ]]; then
    run_supervised 0 "${base_port}" astropy__astropy-14369
  else
    run_supervised 0 "${base_port}" astropy__astropy-14995
  fi
}

run_worker_one() {
  if [[ "${RUN_GROUP}" == "A" ]]; then
    run_supervised 1 "$((base_port + 1))" astropy__astropy-13236
    run_supervised 1 "$((base_port + 1))" astropy__astropy-14096
  else
    run_supervised 1 "$((base_port + 1))" astropy__astropy-14508
  fi
}

run_worker_zero &
worker_zero_pid=$!
run_worker_one &
worker_one_pid=$!
wait "${worker_zero_pid}"
wait "${worker_one_pid}"
