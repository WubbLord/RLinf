#!/usr/bin/env bash
#SBATCH --job-name=envreward-eval-gs500
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/data/scratch/zhangbr/RLinf/slurm/logs/%x-%j.out
#SBATCH --error=/data/scratch/zhangbr/RLinf/slurm/logs/%x-%j.err
#SBATCH --chdir=/data/scratch/zhangbr/RLinf

set -euo pipefail

REPO_DIR="/data/scratch/zhangbr/RLinf"
cd "${REPO_DIR}"

mkdir -p slurm/logs logs/eval logs/exported_full_weights

export PYTHONPATH="${PYTHONPATH:-}"
source .venv-roboreward/bin/activate

export EMBODIED_PATH="${REPO_DIR}/examples/embodiment"
export REPO_PATH="${REPO_DIR}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/path/to/RoboTwin}"
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export SCRATCH="${SCRATCH:-/data/scratch/zhangbr}"
export WANDB_DIR="${WANDB_DIR:-${REPO_DIR}/logs/wandb}"
export RAY_TMPDIR="${RAY_TMPDIR:-${SCRATCH}/ray_tmp/${SLURM_JOB_ID:-envreward-eval}}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-180}"
mkdir -p "${WANDB_DIR}" "${RAY_TMPDIR}"

set -a
source workspace.env
set +a

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi

CKPT_DIR="${CKPT_DIR:-${REPO_DIR}/logs/20260522-20:51:10-maniskill_grpo_openvlaoft/maniskill_grpo_openvlaoft/checkpoints/global_step_500}"
EXPORT_DIR="${EXPORT_DIR:-${REPO_DIR}/logs/exported_full_weights/envreward_global_step_500}"
FULL_WEIGHTS="${FULL_WEIGHTS:-${CKPT_DIR}/actor/model_state_dict/full_weights.pt}"
RUN_TAG="${RUN_TAG:-$(date +'%Y%m%d-%H%M%S')-envreward-gs500}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${REPO_DIR}/logs/eval/${RUN_TAG}}"

EVAL_EXTRA_OVERRIDES=()
for override in "$@"; do
  case "${override}" in
    env.eval.video_cfg.source=*|+env.eval.video_cfg.source=*|++env.eval.video_cfg.source=*)
      video_source="${override#*=}"
      # `++` works when the key is absent or already present in the config.
      EVAL_EXTRA_OVERRIDES+=("++env.eval.video_cfg.source=${video_source}")
      ;;
    *)
      EVAL_EXTRA_OVERRIDES+=("${override}")
      ;;
  esac
done

NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
if [[ ! "${NUM_GPUS}" =~ ^[0-9]+$ ]]; then
  NUM_GPUS=1
fi

JOB_ID_FOR_PORTS="${SLURM_JOB_ID:-$$}"
PORT_BLOCK=$((JOB_ID_FOR_PORTS % 40))
RAY_PORT_BASE=$((20000 + PORT_BLOCK * 1000))
RAY_HEAD_PORT="${RAY_HEAD_PORT:-${RAY_PORT_BASE}}"
RAY_NODE_MANAGER_PORT="${RAY_NODE_MANAGER_PORT:-$((RAY_PORT_BASE + 1))}"
RAY_OBJECT_MANAGER_PORT="${RAY_OBJECT_MANAGER_PORT:-$((RAY_PORT_BASE + 2))}"
RAY_CLIENT_SERVER_PORT="${RAY_CLIENT_SERVER_PORT:-$((RAY_PORT_BASE + 3))}"
RAY_RUNTIME_ENV_AGENT_PORT="${RAY_RUNTIME_ENV_AGENT_PORT:-$((RAY_PORT_BASE + 4))}"
RAY_DASHBOARD_AGENT_LISTEN_PORT="${RAY_DASHBOARD_AGENT_LISTEN_PORT:-$((RAY_PORT_BASE + 5))}"
RAY_DASHBOARD_AGENT_GRPC_PORT="${RAY_DASHBOARD_AGENT_GRPC_PORT:-$((RAY_PORT_BASE + 6))}"
RAY_METRICS_EXPORT_PORT="${RAY_METRICS_EXPORT_PORT:-$((RAY_PORT_BASE + 7))}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-$((RAY_PORT_BASE + 100))}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-$((RAY_PORT_BASE + 899))}"

RAY_START_LOG="${RAY_TMPDIR}/ray-start-${SLURM_JOB_ID:-manual}.log"
echo "Using Ray port range ${RAY_PORT_BASE}-$((RAY_PORT_BASE + 999))"
unset RAY_ADDRESS
python -m ray.scripts.scripts start --head \
  --port="${RAY_HEAD_PORT}" \
  --include-dashboard=false \
  --dashboard-port=0 \
  --node-manager-port="${RAY_NODE_MANAGER_PORT}" \
  --object-manager-port="${RAY_OBJECT_MANAGER_PORT}" \
  --ray-client-server-port="${RAY_CLIENT_SERVER_PORT}" \
  --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}" \
  --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
  --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
  --metrics-export-port="${RAY_METRICS_EXPORT_PORT}" \
  --min-worker-port="${RAY_MIN_WORKER_PORT}" \
  --max-worker-port="${RAY_MAX_WORKER_PORT}" \
  --num-gpus="${NUM_GPUS}" \
  --num-cpus="${SLURM_CPUS_PER_TASK:-32}" \
  --disable-usage-stats \
  --temp-dir="${RAY_TMPDIR}" 2>&1 | tee "${RAY_START_LOG}"

RAY_ADDRESS="$(python - "${RAY_START_LOG}" <<'PY'
import re
import sys

text = open(sys.argv[1], errors="ignore").read()
text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
match = re.search(r"ray start --address='([^']+)'", text)
if match is None:
    match = re.search(r"ray start --address=([^\s]+)", text)
if match is None:
    raise SystemExit("Could not parse Ray address from ray start output")
print(match.group(1))
PY
)"
export RAY_ADDRESS
echo "Using RAY_ADDRESS=${RAY_ADDRESS}"

if [[ ! -f "${FULL_WEIGHTS}" ]]; then
  export EXPORT_RESUME_DIR="${CKPT_DIR}"
  export EXPORT_OUTPUT_DIR="${EXPORT_DIR}"
  FULL_WEIGHTS="${EXPORT_DIR}/actor/model_state_dict/full_weights.pt"
  python examples/embodiment/export_fsdp_full_weights.py \
    --config-path "${EMBODIED_PATH}/config" \
    --config-name maniskill_grpo_openvlaoft
  rm -rf "${EXPORT_DIR}/actor/dcp_checkpoint"
else
  echo "Using existing full weights at ${FULL_WEIGHTS}"
fi
test -f "${FULL_WEIGHTS}"

python examples/embodiment/eval_roboreward_rollout.py \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name maniskill_grpo_openvlaoft_roboreward \
  "runner.ckpt_path=${FULL_WEIGHTS}" \
  "runner.logger.log_path=${EVAL_LOG_DIR}" \
  "runner.logger.experiment_name=envreward_gs500_eval" \
  "env.eval.video_cfg.save_video=True" \
  "env.eval.video_cfg.info_on_video=True" \
  "env.eval.video_cfg.video_base_dir=${EVAL_LOG_DIR}/video" \
  "${EVAL_EXTRA_OVERRIDES[@]}"

echo "Full weights: ${FULL_WEIGHTS}"
echo "Eval logs: ${EVAL_LOG_DIR}"
