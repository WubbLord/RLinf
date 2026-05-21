# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from typing import Any

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.scheduler import Cluster
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _save_raw_outputs(results: list[dict | None], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for rank, result in enumerate(results):
        if result is None:
            continue
        torch.save(result, os.path.join(output_dir, f"env_rank_{rank}.pt"))
        with open(os.path.join(output_dir, f"env_rank_{rank}.json"), "w") as f:
            json.dump(_json_value(result), f, indent=2)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="maniskill_grpo_openvlaoft_roboreward",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    if not cfg.reward.get("use_reward_model", False):
        raise ValueError("reward.use_reward_model must be true for RoboReward eval.")
    if cfg.runner.get("ckpt_path", None) is None:
        raise ValueError("runner.ckpt_path must point to exported full_weights.pt.")

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=component_placement.get_strategy("rollout"),
    )
    env_group = EnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=component_placement.get_strategy("env"),
    )
    reward_group = EmbodiedRewardWorker.create_group(cfg).launch(
        cluster,
        name=cfg.reward.group_name,
        placement_strategy=component_placement.get_strategy("reward"),
    )

    rollout_handle = rollout_group.init_worker()
    env_handle = env_group.init_worker()
    reward_group.init_worker().wait()
    rollout_handle.wait()
    env_handle.wait()

    env_channel = Channel.create("Env")
    rollout_channel = Channel.create("Rollout")
    reward_channel = Channel.create("Reward")

    env_run_handle: Handle = env_group.interact(
        input_channel=env_channel,
        rollout_channel=rollout_channel,
        reward_channel=reward_channel,
        actor_channel=None,
    )
    rollout_run_handle: Handle = rollout_group.generate(
        input_channel=rollout_channel,
        output_channel=env_channel,
    )
    reward_run_handle: Handle = reward_group.compute_rewards(
        input_channel=reward_channel,
        output_channel=env_channel,
    )

    env_results = env_run_handle.wait()
    rollout_run_handle.wait()
    reward_run_handle.wait()

    raw_output_dir = os.path.join(cfg.runner.logger.log_path, "roboreward_outputs")
    _save_raw_outputs(env_results, raw_output_dir)

    env_metrics_list = [result for result in env_results if result is not None]
    metrics = compute_evaluate_metrics(env_metrics_list)
    metrics = {f"roboreward_rollout/{key}": value for key, value in metrics.items()}

    with open(os.path.join(raw_output_dir, "summary.json"), "w") as f:
        json.dump(_json_value(metrics), f, indent=2)

    logger = get_logger()
    logger.info(metrics)
    metric_logger = MetricLogger(cfg)
    metric_logger.log(data=metrics, step=0)
    metric_logger.finish()


if __name__ == "__main__":
    main()
