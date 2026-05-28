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

import csv
import json
import os
from collections import Counter
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
from rlinf.workers.env.env_worker import get_env_worker_class
from rlinf.workers.reward.reward_worker import get_embodied_reward_worker_class
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)

ROBOREWARD_OUTPUT_KEYS = ("roboreward_terminal", "reward_model_output")


def _format_eval_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    formatted = {}
    for key, value in metrics.items():
        formatted[f"roboreward_rollout/{key}"] = value
        if key not in ROBOREWARD_OUTPUT_KEYS:
            formatted[f"eval/{key}"] = value
    return formatted


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


def _tensor_to_scalar(value: torch.Tensor, index: int) -> Any:
    item = value[index].detach().cpu().reshape(-1)
    if item.numel() == 0:
        return None
    if item.numel() == 1:
        return item.item()
    return item.tolist()


def _format_list(values: list[Any]) -> str:
    return " ".join(str(value) for value in values)


def _infer_raw_score(normalized_value: float) -> int:
    # RoboReward normalized score mapping is (raw_score - 1) / 4.
    return int(round(normalized_value * 4 + 1))


def _rollout_count(result: dict[str, Any]) -> int:
    for key in ("success_once", "return", "episode_len", "reward"):
        value = result.get(key)
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])

    rollout_count = 0
    for key, value in result.items():
        if key in ROBOREWARD_OUTPUT_KEYS:
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            rollout_count = max(rollout_count, int(value.shape[0]))
    return rollout_count


def _reward_model_outputs(
    result: dict[str, Any], rollout_count: int
) -> tuple[torch.Tensor, bool] | None:
    value = None
    for key in ROBOREWARD_OUTPUT_KEYS:
        candidate = result.get(key)
        if isinstance(candidate, torch.Tensor):
            value = candidate
            break
    if not isinstance(value, torch.Tensor) or rollout_count <= 0:
        return None
    value = value.detach().cpu()
    if value.numel() == rollout_count:
        return value.reshape(rollout_count, 1), False
    if value.ndim == 1:
        if value.numel() % rollout_count != 0:
            return None
        return value.reshape(rollout_count, -1), True
    if value.shape[0] == rollout_count:
        outputs = value.reshape(rollout_count, -1)
        return outputs, outputs.shape[1] > 1
    return None


def _save_per_rollout_outputs(
    results: list[dict | None], output_dir: str, cfg
) -> None:
    rows: list[dict[str, Any]] = []
    global_rollout_idx = 0
    eval_num_envs = int(cfg.env.eval.total_num_envs)
    for rank, result in enumerate(results):
        if result is None:
            continue
        rollout_count = _rollout_count(result)
        reward_model_result = _reward_model_outputs(result, rollout_count)
        reward_model_outputs = (
            reward_model_result[0] if reward_model_result is not None else None
        )
        reward_model_has_steps = (
            reward_model_result[1] if reward_model_result is not None else False
        )
        for local_idx in range(rollout_count):
            env_index = local_idx % eval_num_envs
            rollout_epoch = local_idx // eval_num_envs
            row: dict[str, Any] = {
                "rollout_index": global_rollout_idx,
                "env_rank": rank,
                "rank_rollout_index": local_idx,
                "rollout_epoch": rollout_epoch,
                "env_index": env_index,
            }
            for key, value in result.items():
                if (
                    not isinstance(value, torch.Tensor)
                    or value.ndim == 0
                    or key in ROBOREWARD_OUTPUT_KEYS
                    or value.shape[0] <= local_idx
                ):
                    continue
                row[key] = _tensor_to_scalar(value, local_idx)
            if reward_model_outputs is not None:
                rewards = reward_model_outputs[local_idx]
                nonzero_steps = (rewards != 0).nonzero(as_tuple=False).reshape(-1)
                terminal_values = [float(rewards[idx].item()) for idx in nonzero_steps]
                terminal_steps = (
                    [int(idx.item()) for idx in nonzero_steps]
                    if reward_model_has_steps
                    else []
                )
                row["roboreward_terminal_sum_normalized"] = sum(terminal_values)
                row["roboreward_terminal_event_count"] = len(terminal_values)
                row["roboreward_terminal_steps"] = terminal_steps
                row["roboreward_terminal_values_normalized"] = terminal_values
                row["roboreward_raw_scores_inferred"] = [
                    _infer_raw_score(value) for value in terminal_values
                ]
                row["has_multiple_roboreward_events"] = len(terminal_values) > 1
            rows.append(row)
            global_rollout_idx += 1

    with open(os.path.join(output_dir, "per_rollout_outputs.json"), "w") as f:
        json.dump(rows, f, indent=2)
    _save_per_rollout_csvs(rows, output_dir, eval_num_envs)


def _save_per_rollout_csvs(
    rows: list[dict[str, Any]], output_dir: str, eval_num_envs: int
) -> None:
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        csv_rows.append(
            {
                "rollout_index": row["rollout_index"],
                "rollout_epoch": row["rollout_epoch"],
                "env_index": row["env_index"],
                "success_once": bool(row.get("success_once", False)),
                "env_return": row.get("return"),
                "env_reward_per_step": row.get("reward"),
                "episode_len": row.get("episode_len"),
                "roboreward_terminal_sum_normalized": row.get(
                    "roboreward_terminal_sum_normalized", 0
                ),
                "roboreward_terminal_event_count": row.get(
                    "roboreward_terminal_event_count", 0
                ),
                "roboreward_terminal_steps": _format_list(
                    row.get("roboreward_terminal_steps", [])
                ),
                "roboreward_terminal_values_normalized": _format_list(
                    row.get("roboreward_terminal_values_normalized", [])
                ),
                "roboreward_raw_scores_inferred": _format_list(
                    row.get("roboreward_raw_scores_inferred", [])
                ),
                "has_multiple_roboreward_events": bool(
                    row.get("has_multiple_roboreward_events", False)
                ),
            }
        )

    per_rollout_csv = os.path.join(output_dir, "per_rollout_rewards.csv")
    fieldnames = [
        "rollout_index",
        "rollout_epoch",
        "env_index",
        "success_once",
        "env_return",
        "env_reward_per_step",
        "episode_len",
        "roboreward_terminal_sum_normalized",
        "roboreward_terminal_event_count",
        "roboreward_terminal_steps",
        "roboreward_terminal_values_normalized",
        "roboreward_raw_scores_inferred",
        "has_multiple_roboreward_events",
    ]
    with open(per_rollout_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    epochs = sorted({int(row["rollout_epoch"]) for row in csv_rows})
    rows_by_env_epoch = {
        (int(row["env_index"]), int(row["rollout_epoch"])): row for row in csv_rows
    }
    pivot_fieldnames = ["env_index"]
    for epoch in epochs:
        prefix = f"epoch{epoch}"
        pivot_fieldnames.extend(
            [
                f"{prefix}_success_once",
                f"{prefix}_env_return",
                f"{prefix}_roboreward_sum_norm",
                f"{prefix}_roboreward_event_count",
                f"{prefix}_roboreward_raw_scores_inferred",
            ]
        )

    pivot_rows = []
    for env_index in range(eval_num_envs):
        pivot_row: dict[str, Any] = {"env_index": env_index}
        for epoch in epochs:
            row = rows_by_env_epoch.get((env_index, epoch), {})
            prefix = f"epoch{epoch}"
            pivot_row[f"{prefix}_success_once"] = row.get("success_once", "")
            pivot_row[f"{prefix}_env_return"] = row.get("env_return", "")
            pivot_row[f"{prefix}_roboreward_sum_norm"] = row.get(
                "roboreward_terminal_sum_normalized", ""
            )
            pivot_row[f"{prefix}_roboreward_event_count"] = row.get(
                "roboreward_terminal_event_count", ""
            )
            pivot_row[f"{prefix}_roboreward_raw_scores_inferred"] = row.get(
                "roboreward_raw_scores_inferred", ""
            )
        pivot_rows.append(pivot_row)

    pivot_csv = os.path.join(output_dir, "per_env_epoch_pivot.csv")
    with open(pivot_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pivot_fieldnames)
        writer.writeheader()
        writer.writerows(pivot_rows)

    event_counts = Counter(
        int(row["roboreward_terminal_event_count"]) for row in csv_rows
    )
    terminal_sums = Counter(
        str(row["roboreward_terminal_sum_normalized"]) for row in csv_rows
    )
    summary = {
        "per_rollout_csv": per_rollout_csv,
        "per_env_epoch_pivot_csv": pivot_csv,
        "num_rollout_windows": len(csv_rows),
        "num_epochs": len(epochs),
        "num_envs": eval_num_envs,
        "success_rate": float(
            np.mean([bool(row["success_once"]) for row in csv_rows])
        )
        if csv_rows
        else None,
        "mean_env_return": float(
            np.mean([float(row["env_return"]) for row in csv_rows])
        )
        if csv_rows
        else None,
        "mean_env_reward_per_step": float(
            np.mean([float(row["env_reward_per_step"]) for row in csv_rows])
        )
        if csv_rows
        else None,
        "mean_roboreward_terminal_sum_normalized": float(
            np.mean(
                [
                    float(row["roboreward_terminal_sum_normalized"])
                    for row in csv_rows
                ]
            )
        )
        if csv_rows
        else None,
        "roboreward_terminal_event_count_distribution": dict(sorted(event_counts.items())),
        "roboreward_terminal_sum_distribution": dict(sorted(terminal_sums.items())),
        "windows_with_multiple_roboreward_events": sum(
            bool(row["has_multiple_roboreward_events"]) for row in csv_rows
        ),
        "note": (
            "RoboReward values are normalized from raw scores by (score-1)/4. "
            "Raw integer scores are inferred from normalized values; decoded model "
            "text is not saved."
        ),
    }
    with open(os.path.join(output_dir, "per_rollout_rewards_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="maniskill_grpo_openvlaoft_roboreward",
)
def main(cfg) -> None:
    cfg.runner.only_eval = True
    cfg.env.eval.auto_reset = False
    cfg.env.eval.ignore_terminations = False
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
    env_worker_cls = get_env_worker_class(cfg)
    env_group = env_worker_cls.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=component_placement.get_strategy("env"),
    )
    reward_worker_cls = get_embodied_reward_worker_class(cfg)
    reward_group = reward_worker_cls.create_group(cfg).launch(
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

    env_run_handle: Handle = env_group.evaluate_with_reward_model(
        input_channel=env_channel,
        rollout_channel=rollout_channel,
        reward_channel=reward_channel,
    )
    rollout_run_handle: Handle = rollout_group.evaluate(
        input_channel=rollout_channel,
        output_channel=env_channel,
    )
    reward_run_handle: Handle = reward_group.compute_rewards(
        input_channel=reward_channel,
        output_channel=env_channel,
        mode="eval",
    )

    env_results = env_run_handle.wait()
    rollout_run_handle.wait()
    reward_run_handle.wait()

    raw_output_dir = os.path.join(cfg.runner.logger.log_path, "roboreward_outputs")
    _save_raw_outputs(env_results, raw_output_dir)
    _save_per_rollout_outputs(env_results, raw_output_dir, cfg)

    env_metrics_list = [result for result in env_results if result is not None]
    metrics = compute_evaluate_metrics(env_metrics_list)
    metrics = _format_eval_metrics(metrics)

    with open(os.path.join(raw_output_dir, "summary.json"), "w") as f:
        json.dump(_json_value(metrics), f, indent=2)
    with open(os.path.join(cfg.runner.logger.log_path, "summary.json"), "w") as f:
        json.dump(_json_value(metrics), f, indent=2)

    logger = get_logger()
    logger.info(metrics)
    metric_logger = MetricLogger(cfg)
    metric_logger.log(data=metrics, step=0)
    metric_logger.finish()


if __name__ == "__main__":
    main()
