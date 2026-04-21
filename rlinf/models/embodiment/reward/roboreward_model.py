# Copyright 2026 The RLinf Authors.
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

"""RoboReward model wrapper for embodied RL inference."""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.config import torch_dtype_from_precision
from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel


class RoboRewardModel(BaseRewardModel):
    """Qwen3-VL-based rollout-video reward model.

    This model wraps the RoboReward checkpoint, which expects a task description
    and a rollout video and returns a discrete progress score in ``[1, 5]``.
    """

    DEFAULT_PROMPT_TEMPLATE = (
        "Given the task, assign a discrete progress score reward (1,2,3,4,5) for "
        "the robot in the video in the format: ANSWER: <score>\n"
        "Rubric for end-of-episode progress (judge only the final state without time limits):\n"
        "1 - No Success: Final state shows no goal-relevant change for the command.\n"
        "2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.\n"
        "3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement.\n"
        "4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.\n"
        "5 - Perfect Completion: Final state satisfies all requirements.\n\n"
        "Task: {task_description}"
    )

    SCORE_RE = re.compile(r"ANSWER:\s*([1-5])", re.IGNORECASE)
    FALLBACK_SCORE_RE = re.compile(r"\b([1-5])\b")

    def __init__(self, cfg: DictConfig):
        """Initialize RoboReward inference model."""
        super().__init__(cfg)
        self.cfg = cfg
        self.model_path = cfg.model_path
        self.max_new_tokens = int(cfg.get("max_new_tokens", 16))
        self.do_sample = bool(cfg.get("do_sample", False))
        self.score_mapping = cfg.get("score_mapping", "normalized")
        self.prompt_template = cfg.get("prompt_template", self.DEFAULT_PROMPT_TEMPLATE)
        self.trust_remote_code = bool(cfg.get("trust_remote_code", True))
        self.low_cpu_mem_usage = bool(cfg.get("low_cpu_mem_usage", True))
        self.attn_implementation = cfg.get("attn_implementation", None)
        self.torch_dtype = torch_dtype_from_precision(cfg.get("precision", "bf16"))

        self.processor, self.backbone = self._load_hf_components()

    def _load_hf_components(self):
        """Load processor and model with compatibility fallbacks."""
        try:
            import transformers
            from transformers import AutoProcessor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "RoboReward requires the `transformers` package."
            ) from exc

        processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )

        model_loaders = []
        for loader_name in (
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        ):
            loader = getattr(transformers, loader_name, None)
            if loader is not None:
                model_loaders.append(loader)

        load_errors: list[Exception] = []
        for loader in model_loaders:
            try:
                kwargs = {
                    "trust_remote_code": self.trust_remote_code,
                    "low_cpu_mem_usage": self.low_cpu_mem_usage,
                    "torch_dtype": self.torch_dtype,
                }
                if self.attn_implementation is not None:
                    kwargs["attn_implementation"] = self.attn_implementation
                return processor, loader.from_pretrained(self.model_path, **kwargs)
            except Exception as exc:  # pragma: no cover - loader availability varies
                load_errors.append(exc)

        if load_errors:
            last_error = load_errors[-1]
            raise ImportError(
                "Unable to load RoboReward-8B. The installed `transformers` "
                "package does not recognize `qwen3_vl`. Install a Transformers "
                "version with Qwen3-VL support and retry."
            ) from last_error
        raise ImportError("No compatible HuggingFace auto model loader was found.")

    def to(self, *args, **kwargs):
        """Move wrapped HuggingFace model together with the module shell."""
        super().to(*args, **kwargs)
        self.backbone = self.backbone.to(*args, **kwargs)
        return self

    def eval(self):
        """Switch wrapped model to eval mode."""
        super().eval()
        self.backbone.eval()
        return self

    def train(self, mode: bool = True):
        """Mirror training mode to the wrapped model."""
        super().train(mode)
        self.backbone.train(mode)
        return self

    def forward(
        self,
        input_data: Any,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Forward pass is not implemented for reward-model training."""
        raise NotImplementedError(
            "RoboRewardModel currently supports inference-only embodied reward use."
        )

    @classmethod
    def parse_reward_score(cls, text: str) -> int:
        """Extract a progress score from model output text."""
        match = cls.SCORE_RE.search(text)
        if match is not None:
            return int(match.group(1))

        fallback_match = cls.FALLBACK_SCORE_RE.search(text)
        if fallback_match is not None:
            return int(fallback_match.group(1))
        raise ValueError(f"Could not parse RoboReward score from output: {text!r}")

    @staticmethod
    def map_scores_to_rewards(
        scores: torch.Tensor,
        mapping: str,
    ) -> torch.Tensor:
        """Map raw integer RoboReward scores to scalar RL rewards."""
        if mapping == "raw":
            return scores
        if mapping == "normalized":
            return (scores - 1.0) / 4.0
        if mapping == "centered":
            return (scores - 3.0) / 2.0
        raise ValueError(
            f"Unsupported RoboReward score mapping {mapping!r}. "
            "Supported mappings: raw, normalized, centered."
        )

    def _format_prompt(self, task_description: str) -> str:
        task_description = str(task_description).strip() or "unknown task"
        return self.prompt_template.format(task_description=task_description)

    @staticmethod
    def _to_numpy_video(video: Any) -> np.ndarray:
        """Convert one rollout video to ``[T, H, W, C]`` numpy format."""
        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().numpy()
        else:
            video = np.asarray(video)

        if video.ndim != 4:
            raise ValueError(
                f"Expected video with shape [T, H, W, C], but got {video.shape}."
            )
        if video.dtype != np.uint8:
            video = np.clip(video, 0, 255).astype(np.uint8)
        return video

    def _prepare_batch_inputs(
        self,
        videos: list[np.ndarray],
        task_descriptions: list[str],
    ) -> dict[str, torch.Tensor]:
        prompts = []
        for task_description in task_descriptions:
            prompts.append(self._format_prompt(task_description))

        chat_texts = []
        for prompt in prompts:
            if hasattr(self.processor, "apply_chat_template"):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": "rollout"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                chat_texts.append(
                    self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            else:  # pragma: no cover
                chat_texts.append(prompt)

        inputs = self.processor(
            text=chat_texts,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def compute_reward(
        self,
        observations: dict[str, Any],
        task_descriptions: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Run RoboReward inference on a batch of rollout videos."""
        if not isinstance(observations, dict):
            raise TypeError(
                "RoboRewardModel expects a dict with `videos` and `task_descriptions`."
            )
        if "videos" not in observations:
            raise KeyError("RoboRewardModel observations must contain `videos`.")

        videos_input = observations["videos"]
        if isinstance(videos_input, torch.Tensor):
            videos = [self._to_numpy_video(video) for video in videos_input]
        else:
            videos = [self._to_numpy_video(video) for video in videos_input]

        if task_descriptions is None:
            task_descriptions = observations.get("task_descriptions")
        if task_descriptions is None:
            raise KeyError(
                "RoboRewardModel requires `task_descriptions` for reward inference."
            )
        task_descriptions = [
            str(task_description) for task_description in task_descriptions
        ]

        inputs = self._prepare_batch_inputs(videos, task_descriptions)
        device = next(self.backbone.parameters()).device
        inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            generated_ids = self.backbone.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
            )

        prompt_length = inputs["input_ids"].shape[1]
        trimmed_ids = generated_ids[:, prompt_length:]
        decoded = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        scores = torch.tensor(
            [self.parse_reward_score(text) for text in decoded],
            device=device,
            dtype=torch.float32,
        )
        rewards = self.map_scores_to_rewards(scores, self.score_mapping)
        return rewards
