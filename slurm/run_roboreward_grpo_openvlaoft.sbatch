#!/usr/bin/env bash
#SBATCH --job-name=roboreward-grpo-openvlaoft
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/data/scratch/zhangbr/RLinf/slurm/logs/%x-%j.out
#SBATCH --error=/data/scratch/zhangbr/RLinf/slurm/logs/%x-%j.err
#SBATCH --chdir=/data/scratch/zhangbr/RLinf

set -euo pipefail

REPO_DIR="/data/scratch/zhangbr/RLinf"
cd "${REPO_DIR}"

mkdir -p slurm/logs logs

source .venv-roboreward/bin/activate

export PYTHONUNBUFFERED=1
export SCRATCH="${SCRATCH:-/data/scratch/zhangbr}"
export WANDB_DIR="${WANDB_DIR:-${REPO_DIR}/logs/wandb}"
export RAY_TMPDIR="${RAY_TMPDIR:-${SCRATCH}/ray_tmp/${SLURM_JOB_ID:-roboreward-grpo}}"
mkdir -p "${WANDB_DIR}" "${RAY_TMPDIR}"

set -a
source workspace.env
set +a

ray stop --force >/dev/null 2>&1 || true

NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
if [[ ! "${NUM_GPUS}" =~ ^[0-9]+$ ]]; then
  NUM_GPUS=1
fi

ray start --head \
  --num-gpus="${NUM_GPUS}" \
  --num-cpus="${SLURM_CPUS_PER_TASK:-32}" \
  --temp-dir="${RAY_TMPDIR}"

cleanup() {
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

bash examples/embodiment/run_embodiment.sh maniskill_grpo_openvlaoft_roboreward
