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
import re

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, open_dict

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement

mp.set_start_method("spawn", force=True)


def _get_actor_worker_cls(cfg):
    if cfg.algorithm.loss_type == "embodied_sac":
        from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

        return EmbodiedSACFSDPPolicy
    if cfg.algorithm.loss_type == "embodied_dagger":
        from rlinf.workers.actor.fsdp_dagger_policy_worker import (
            EmbodiedDAGGERFSDPPolicy,
        )

        return EmbodiedDAGGERFSDPPolicy
    if cfg.algorithm.loss_type == "embodied_nft":
        from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy

        return EmbodiedNFTFSDPPolicy

    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

    return EmbodiedFSDPActor


def _step_from_resume_dir(resume_dir: str) -> int:
    match = re.search(r"global_step_(\d+)", resume_dir)
    if match is None:
        return 0
    return int(match.group(1))


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="maniskill_grpo_openvlaoft_roboreward",
)
def main(cfg) -> None:
    resume_dir = os.environ.get("EXPORT_RESUME_DIR") or cfg.runner.get(
        "resume_dir", None
    )
    output_dir = os.environ.get("EXPORT_OUTPUT_DIR")
    overwrite = os.environ.get("EXPORT_OVERWRITE", "0") == "1"

    if resume_dir is None:
        raise ValueError("Set EXPORT_RESUME_DIR or runner.resume_dir.")
    resume_dir = os.path.abspath(resume_dir)
    if output_dir is None:
        output_dir = os.path.join(resume_dir, "exported_full_weights")
    output_dir = os.path.abspath(output_dir)

    actor_checkpoint_path = os.path.join(resume_dir, "actor")
    full_weights_path = os.path.join(
        output_dir, "actor", "model_state_dict", "full_weights.pt"
    )
    if os.path.exists(full_weights_path) and not overwrite:
        print(f"Full weights already exist at {full_weights_path}")
        return
    if not os.path.isdir(actor_checkpoint_path):
        raise FileNotFoundError(f"Missing actor checkpoint: {actor_checkpoint_path}")

    with open_dict(cfg):
        cfg.actor.fsdp_config.save_full_model_weights = True
        cfg.runner.resume_dir = resume_dir
        cfg.runner.logger.log_path = os.path.join(output_dir, "export_logs")
        cfg.runner.logger.logger_backends = []

    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)
    actor_placement = component_placement.get_strategy("actor")
    actor_worker_cls = _get_actor_worker_cls(cfg)
    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    actor_group.init_worker().wait()
    print(f"Loading DCP actor checkpoint from {actor_checkpoint_path}")
    actor_group.load_checkpoint(actor_checkpoint_path).wait()

    export_actor_path = os.path.join(output_dir, "actor")
    os.makedirs(export_actor_path, exist_ok=True)
    print(f"Exporting full actor weights to {full_weights_path}")
    actor_group.save_checkpoint(export_actor_path, _step_from_resume_dir(resume_dir)).wait()

    if not os.path.exists(full_weights_path):
        raise FileNotFoundError(f"Export did not create {full_weights_path}")
    print(f"Exported full weights: {full_weights_path}")


if __name__ == "__main__":
    main()
