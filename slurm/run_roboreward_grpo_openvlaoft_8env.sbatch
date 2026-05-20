#!/usr/bin/env bash
#SBATCH --job-name=roboreward-grpo-8env-full
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/data/scratch/zhangbr/RLinf/slurm/logs/%x-%j.out
#SBATCH --error=/data/scratch/zhangbr/RLinf/slurm/logs/%x-%j.err
#SBATCH --chdir=/data/scratch/zhangbr/RLinf

set -euo pipefail

REPO_DIR="/data/scratch/zhangbr/RLinf"
cd "${REPO_DIR}"

mkdir -p slurm/logs logs

export PYTHONPATH="${PYTHONPATH:-}"
source .venv-roboreward/bin/activate

export EMBODIED_PATH="${REPO_DIR}/examples/embodiment"
export REPO_PATH="${REPO_DIR}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/path/to/RoboTwin}"
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export SCRATCH="${SCRATCH:-/data/scratch/zhangbr}"
export WANDB_DIR="${WANDB_DIR:-${REPO_DIR}/logs/wandb}"
export RAY_TMPDIR="${RAY_TMPDIR:-${SCRATCH}/ray_tmp/${SLURM_JOB_ID:-roboreward-grpo-8env-full}}"
mkdir -p "${WANDB_DIR}" "${RAY_TMPDIR}"

set -a
source workspace.env
set +a

if [[ -f "${REPO_DIR}/.env" ]]; then
  WANDB_API_KEY_FROM_ENV="$(grep -E '^WANDB_API_KEY=' "${REPO_DIR}/.env" | tail -1 | cut -d'=' -f2- || true)"
  if [[ -n "${WANDB_API_KEY_FROM_ENV}" ]]; then
    export WANDB_API_KEY="${WANDB_API_KEY_FROM_ENV}"
  fi
  unset WANDB_API_KEY_FROM_ENV
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi

ray stop --force >/dev/null 2>&1 || true

ray start --head \
  --port=0 \
  --num-gpus=1 \
  --num-cpus="${SLURM_CPUS_PER_TASK:-8}" \
  --temp-dir="${RAY_TMPDIR}"

cleanup() {
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

LOG_DIR="${REPO_DIR}/logs/$(date +'%Y%m%d-%H:%M:%S')-maniskill_grpo_openvlaoft_roboreward_8env"

python examples/embodiment/train_embodied_agent.py \
  --config-path "${REPO_DIR}/examples/embodiment/config/" \
  --config-name maniskill_grpo_openvlaoft_roboreward \
  runner.logger.log_path="${LOG_DIR}" \
  env.train.total_num_envs=8 \
  env.eval.total_num_envs=8
