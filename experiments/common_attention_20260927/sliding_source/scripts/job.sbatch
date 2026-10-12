#!/usr/bin/env bash
#SBATCH --job-name=tebench
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=ghx4-interactive
#SBATCH --account=bcjw-dtai-gh
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/tebench-%j.log
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?Submit from the project root}/scripts/env.sh"
exec "$AGENT_ENV/bin/python" -B -u -m tebench.job "$@"
