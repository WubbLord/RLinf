#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

# export MUJOCO_GL="osmesa"
# export PYOPENGL_PLATFORM="osmesa"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH=${DREAMZERO_PATH}:$PYTHONPATH

export HYDRA_FULL_ERROR=1

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
else
    CONFIG_NAME=$1
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}

export ROBOT_PLATFORM

# Libero variant: standard, pro, plus
export LIBERO_TYPE=${LIBERO_TYPE:-"standard"}
if [ "$LIBERO_TYPE" == "pro" ]; then
    export LIBERO_PERTURBATION="all"  # all,swap,object,lan
    echo "Evaluation Mode: LIBERO-PRO | Perturbation: $LIBERO_PERTURBATION"
elif [ "$LIBERO_TYPE" == "plus" ]; then
    export LIBERO_SUFFIX="all"
    echo "Evaluation Mode: LIBERO-PLUS | Suffix: $LIBERO_SUFFIX"
else
    echo "Evaluation Mode: Standard LIBERO"
fi

echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

# Check if config yaml uses wandb as a logger backend; if so, load credentials and set env vars
CONFIG_FILE="${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml"
if grep -q "wandb" "${CONFIG_FILE}" 2>/dev/null && \
   grep "logger_backends" "${CONFIG_FILE}" 2>/dev/null | grep -q "wandb"; then
    echo "wandb detected in logger_backends, loading credentials..."
    ENV_FILE="${REPO_PATH}/.env"
    if [ -f "${ENV_FILE}" ]; then
        _WANDB_API_KEY=$(grep -E "^WANDB_API_KEY=" "${ENV_FILE}" | cut -d'=' -f2-)
        if [ -n "${_WANDB_API_KEY}" ]; then
            export WANDB_API_KEY="${_WANDB_API_KEY}"
            echo "WANDB_API_KEY loaded from ${ENV_FILE}"
        else
            echo "Warning: WANDB_API_KEY not found in ${ENV_FILE}"
        fi
        unset _WANDB_API_KEY
    else
        echo "Warning: .env file not found at ${ENV_FILE}"
    fi
    _WANDB_ENTITY=$(grep "wandb_entity:" "${CONFIG_FILE}" | awk '{print $2}' | tr -d '"')
    _WANDB_PROJECT=$(grep "wandb_project:" "${CONFIG_FILE}" | awk '{print $2}' | tr -d '"')
    [ -n "${_WANDB_ENTITY}" ]  && export WANDB_ENTITY="${_WANDB_ENTITY}"  && echo "WANDB_ENTITY=${WANDB_ENTITY}"
    [ -n "${_WANDB_PROJECT}" ] && export WANDB_PROJECT="${_WANDB_PROJECT}" && echo "WANDB_PROJECT=${WANDB_PROJECT}"
    unset _WANDB_ENTITY _WANDB_PROJECT
fi

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD}
${CMD} 2>&1 | tee ${MEGA_LOG_FILE}
