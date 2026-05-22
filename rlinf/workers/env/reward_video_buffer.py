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

from typing import Any

import numpy as np
import torch


class RewardVideoBuffer:
    """Tracks per-env rollout videos for terminal video reward models."""

    def __init__(
        self,
        *,
        max_frames: int,
        min_frames: int,
        sample_strategy: str = "uniform_keep_last",
    ):
        if max_frames < min_frames:
            raise ValueError(
                "reward.model.max_frames must be greater than or equal to "
                "reward.model.min_frames."
            )
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.sample_strategy = sample_strategy
        self._stage_buffers: list[dict[str, Any]] = []

    @staticmethod
    def _to_numpy_frame(frame: Any) -> np.ndarray | None:
        if frame is None:
            return None
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        else:
            frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    @staticmethod
    def extract_main_images(obs: Any) -> list[np.ndarray | None]:
        if not isinstance(obs, dict):
            return []
        images = obs.get("main_images")
        if images is None:
            return []
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        else:
            images = np.asarray(images)
        return [RewardVideoBuffer._to_numpy_frame(image) for image in images]

    @staticmethod
    def extract_task_descriptions(obs: Any) -> list[str | None]:
        if not isinstance(obs, dict):
            return []
        task_descriptions = obs.get("task_descriptions")
        if task_descriptions is None:
            return []
        if isinstance(task_descriptions, (list, tuple)):
            return [
                None if task_description is None else str(task_description)
                for task_description in task_descriptions
            ]
        return [str(task_descriptions)]

    @staticmethod
    def sample_frames(
        frames: list[np.ndarray],
        *,
        max_frames: int,
        min_frames: int,
        sample_strategy: str = "uniform_keep_last",
    ) -> np.ndarray:
        if len(frames) == 0:
            raise ValueError("Cannot sample a reward video from an empty frame list.")
        if sample_strategy != "uniform_keep_last":
            raise ValueError(
                f"Unsupported reward video sample strategy: {sample_strategy!r}."
            )

        if len(frames) == 1:
            sampled_frames = [frames[0]]
        else:
            sample_count = min(len(frames), max_frames)
            indices = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
            indices[-1] = len(frames) - 1
            sampled_frames = [frames[idx] for idx in indices]

        target_num_frames = max(max_frames, min_frames)
        while len(sampled_frames) < target_num_frames:
            sampled_frames.append(sampled_frames[-1].copy())
        return np.stack(sampled_frames, axis=0)

    def _ensure_stage(self, stage_id: int, batch_size: int):
        while len(self._stage_buffers) <= stage_id:
            self._stage_buffers.append({})
        state = self._stage_buffers[stage_id]
        if state.get("num_envs") == batch_size:
            return
        self._stage_buffers[stage_id] = {
            "num_envs": batch_size,
            "current_videos": [[] for _ in range(batch_size)],
            "current_task_descriptions": [None for _ in range(batch_size)],
            "pending_videos": [None for _ in range(batch_size)],
            "pending_task_descriptions": [None for _ in range(batch_size)],
        }

    def state(self, stage_id: int) -> dict[str, Any]:
        return self._stage_buffers[stage_id]

    def reset(self, stage_id: int, obs: Any):
        frames = self.extract_main_images(obs)
        task_descriptions = self.extract_task_descriptions(obs)
        batch_size = len(frames)
        self._ensure_stage(stage_id, batch_size)
        state = self._stage_buffers[stage_id]
        state["current_videos"] = [
            [frame] if frame is not None else [] for frame in frames
        ]
        state["current_task_descriptions"] = [
            task_descriptions[idx] if idx < len(task_descriptions) else None
            for idx in range(batch_size)
        ]
        state["pending_videos"] = [None for _ in range(batch_size)]
        state["pending_task_descriptions"] = [None for _ in range(batch_size)]

    def update(
        self,
        stage_id: int,
        obs_list: list[Any] | tuple[Any, ...],
        infos_list: list[Any] | tuple[Any, ...],
        chunk_dones: torch.Tensor,
        *,
        auto_reset: bool,
    ):
        if not isinstance(obs_list, (list, tuple)) or len(obs_list) == 0:
            return

        initial_frames = self.extract_main_images(obs_list[0])
        if len(initial_frames) == 0:
            return
        self._ensure_stage(stage_id, len(initial_frames))
        state = self._stage_buffers[stage_id]
        done_so_far = np.zeros(state["num_envs"], dtype=bool)

        for time_idx, step_obs in enumerate(obs_list):
            step_frames = self.extract_main_images(step_obs)
            step_task_descriptions = self.extract_task_descriptions(step_obs)
            step_infos = infos_list[time_idx] if len(infos_list) > time_idx else None
            terminal_obs = (
                step_infos.get("final_observation")
                if isinstance(step_infos, dict)
                else None
            )
            reset_mask = (
                step_infos.get("_final_observation")
                if isinstance(step_infos, dict)
                else None
            )
            terminal_frames = self.extract_main_images(terminal_obs)
            terminal_task_descriptions = self.extract_task_descriptions(terminal_obs)
            if reset_mask is not None:
                if isinstance(reset_mask, torch.Tensor):
                    reset_mask = reset_mask.detach().cpu().numpy().astype(bool)
                else:
                    reset_mask = np.asarray(reset_mask, dtype=bool)
            else:
                reset_mask = np.zeros(state["num_envs"], dtype=bool)

            for env_idx in range(state["num_envs"]):
                step_done = bool(chunk_dones[env_idx, time_idx].item())
                current_frame = (
                    step_frames[env_idx] if env_idx < len(step_frames) else None
                )
                current_task_description = (
                    step_task_descriptions[env_idx]
                    if env_idx < len(step_task_descriptions)
                    else None
                )
                terminal_frame = (
                    terminal_frames[env_idx] if env_idx < len(terminal_frames) else None
                )
                terminal_task_description = (
                    terminal_task_descriptions[env_idx]
                    if env_idx < len(terminal_task_descriptions)
                    else None
                )

                if step_done and not done_so_far[env_idx]:
                    final_frame = terminal_frame
                    if final_frame is None:
                        final_frame = current_frame
                    if final_frame is not None:
                        state["current_videos"][env_idx].append(final_frame)
                    final_task_description = (
                        terminal_task_description
                        or state["current_task_descriptions"][env_idx]
                        or current_task_description
                    )
                    state["current_task_descriptions"][env_idx] = final_task_description
                    state["pending_videos"][env_idx] = self.sample_frames(
                        state["current_videos"][env_idx],
                        max_frames=self.max_frames,
                        min_frames=self.min_frames,
                        sample_strategy=self.sample_strategy,
                    )
                    state["pending_task_descriptions"][env_idx] = final_task_description
                    done_so_far[env_idx] = True

                    if auto_reset and reset_mask[env_idx]:
                        state["current_videos"][env_idx] = (
                            [current_frame] if current_frame is not None else []
                        )
                        state["current_task_descriptions"][env_idx] = (
                            current_task_description or final_task_description
                        )
                    else:
                        state["current_videos"][env_idx] = []
                    continue

                if done_so_far[env_idx] and not auto_reset:
                    continue
                if current_frame is not None:
                    state["current_videos"][env_idx].append(current_frame)
                if current_task_description is not None:
                    state["current_task_descriptions"][env_idx] = (
                        current_task_description
                    )

    def build_input(
        self,
        stage_id: int,
        reward_input_obs: dict[str, Any],
        done_envs: torch.Tensor,
    ) -> dict[str, Any]:
        state = self._stage_buffers[stage_id]
        fallback_frames = self.extract_main_images(reward_input_obs)
        fallback_task_descriptions = self.extract_task_descriptions(reward_input_obs)

        videos: list[torch.Tensor] = []
        task_descriptions: list[str] = []
        num_envs = state["num_envs"]
        for env_idx in range(num_envs):
            pending_video = state["pending_videos"][env_idx]
            if pending_video is None:
                current_frames = state["current_videos"][env_idx]
                if len(current_frames) == 0 and env_idx < len(fallback_frames):
                    fallback_frame = fallback_frames[env_idx]
                    current_frames = (
                        [fallback_frame] if fallback_frame is not None else []
                    )
                if len(current_frames) == 0:
                    raise ValueError(
                        "Failed to build reward video input because no frames were buffered."
                    )
                pending_video = self.sample_frames(
                    current_frames,
                    max_frames=self.max_frames,
                    min_frames=self.min_frames,
                    sample_strategy=self.sample_strategy,
                )

            task_description = (
                state["pending_task_descriptions"][env_idx]
                or state["current_task_descriptions"][env_idx]
                or (
                    fallback_task_descriptions[env_idx]
                    if env_idx < len(fallback_task_descriptions)
                    else None
                )
                or ""
            )
            videos.append(torch.from_numpy(pending_video).to(torch.uint8))
            task_descriptions.append(str(task_description))

            if done_envs[env_idx].item():
                state["pending_videos"][env_idx] = None
                state["pending_task_descriptions"][env_idx] = None

        return {
            "videos": torch.stack(videos, dim=0),
            "task_descriptions": task_descriptions,
        }
