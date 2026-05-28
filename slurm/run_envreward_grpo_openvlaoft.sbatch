#!/usr/bin/env bash
#SBATCH --job-name=envreward-grpo-openvlaoft
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
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
export RAY_TMPDIR="${RAY_TMPDIR:-${SCRATCH}/ray_tmp/${SLURM_JOB_ID:-envreward-grpo}}"
mkdir -p "${WANDB_DIR}" "${RAY_TMPDIR}"

set -a
source workspace.env
set +a

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  if [[ "${WANDB_MODE:-}" == "online" ]]; then
    echo "WANDB_MODE=online requested but WANDB_API_KEY is not set; using offline mode."
  fi
  export WANDB_MODE="offline"
fi

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

cleanup() {
  # Avoid global `ray stop`: shared nodes may also be running another job's Ray.
  :
}
trap cleanup EXIT

CONFIG_NAME="maniskill_grpo_openvlaoft"
EXTRA_OVERRIDES=()
for override in "$@"; do
  case "${override}" in
    runner.resume_dir=*)
      resume_dir="${override#*=}"
      if [[ "${resume_dir}" != "null" && "${resume_dir}" != "None" && -n "${resume_dir}" ]]; then
        resume_dir="$(
          python - "${resume_dir}" "${CONFIG_NAME}" <<'PY'
import re
import sys
from pathlib import Path

resume_dir = Path(sys.argv[1]).expanduser()
config_name = sys.argv[2]
if not resume_dir.is_absolute():
    resume_dir = Path.cwd() / resume_dir
resume_dir = resume_dir.resolve()

if (resume_dir / "actor").is_dir():
    print(resume_dir)
    raise SystemExit(0)

candidate_roots = [
    resume_dir / config_name / "checkpoints",
    resume_dir / "checkpoints",
    resume_dir,
]
candidates = []
for root in candidate_roots:
    if not root.is_dir():
        continue
    for path in root.glob("global_step_*"):
        match = re.fullmatch(r"global_step_(\d+)", path.name)
        if match and (path / "actor").is_dir():
            candidates.append((int(match.group(1)), path))

if not candidates:
    raise SystemExit(
        f"Could not resolve runner.resume_dir={resume_dir} to a checkpoint "
        "directory containing actor/. Pass .../checkpoints/global_step_N."
    )

print(max(candidates, key=lambda item: item[0])[1])
PY
        )"
        echo "Resolved runner.resume_dir=${resume_dir}"
      fi
      EXTRA_OVERRIDES+=("runner.resume_dir=${resume_dir}")
      ;;
    *)
      EXTRA_OVERRIDES+=("${override}")
      ;;
  esac
done

bash examples/embodiment/run_embodiment.sh "${CONFIG_NAME}" "${EXTRA_OVERRIDES[@]}"
