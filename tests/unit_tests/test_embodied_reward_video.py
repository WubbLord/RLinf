import numpy as np
import torch

from rlinf.models.embodiment.reward.roboreward_model import RoboRewardModel
from rlinf.workers.env.reward_video_buffer import RewardVideoBuffer
from rlinf.workers.reward.reward_worker import RoboRewardEmbodiedRewardWorker


def _make_frame(value: int) -> torch.Tensor:
    return torch.full((2, 2, 3), value, dtype=torch.uint8)


def test_sample_reward_video_frames_keeps_last_and_pads():
    frames = [
        _make_frame(1).numpy(),
        _make_frame(2).numpy(),
        _make_frame(3).numpy(),
    ]

    sampled = RewardVideoBuffer.sample_frames(
        frames,
        max_frames=8,
        min_frames=4,
        sample_strategy="uniform_keep_last",
    )

    assert sampled.shape == (8, 2, 2, 3)
    assert np.all(sampled[0] == 1)
    assert np.all(sampled[-1] == 3)


def test_env_worker_rollout_video_state_tracks_completed_episode():
    buffer = RewardVideoBuffer(
        max_frames=4,
        min_frames=2,
        sample_strategy="uniform_keep_last",
    )

    initial_obs = {
        "main_images": torch.stack([_make_frame(10), _make_frame(20)], dim=0),
        "task_descriptions": ["task a", "task b"],
    }
    buffer.reset(0, initial_obs)

    obs_list = [
        {
            "main_images": torch.stack([_make_frame(11), _make_frame(21)], dim=0),
            "task_descriptions": ["task a", "task b"],
        },
        {
            "main_images": torch.stack([_make_frame(12), _make_frame(22)], dim=0),
            "task_descriptions": ["task a", "task b"],
        },
    ]
    infos_list = [
        {},
        {
            "final_observation": {
                "main_images": torch.stack([_make_frame(99), _make_frame(22)], dim=0),
                "task_descriptions": ["task a done", "task b"],
            },
            "_final_observation": torch.tensor([True, False]),
        },
    ]
    chunk_dones = torch.tensor(
        [
            [False, True],
            [False, False],
        ],
        dtype=torch.bool,
    )

    buffer.update(
        0,
        obs_list,
        infos_list,
        chunk_dones,
        auto_reset=True,
    )

    reward_input_obs = {
        "main_images": torch.stack([_make_frame(12), _make_frame(22)], dim=0),
        "task_descriptions": ["task a next", "task b"],
    }
    reward_input = buffer.build_input(
        stage_id=0,
        reward_input_obs=reward_input_obs,
        done_envs=torch.tensor([True, False], dtype=torch.bool),
    )

    assert reward_input["videos"].shape == (2, 4, 2, 2, 3)
    assert reward_input["task_descriptions"][0] == "task a done"
    assert reward_input["task_descriptions"][1] == "task b"

    state = buffer.state(0)
    assert state["pending_videos"][0] is None
    assert state["pending_task_descriptions"][0] is None
    assert np.all(state["current_videos"][0][0] == 12)


def test_reward_input_merge_and_batch_size():
    payloads = [
        {
            "videos": torch.zeros((1, 2, 2, 2, 3), dtype=torch.uint8),
            "task_descriptions": ["task a"],
            "last_run": torch.ones((1, 1), dtype=torch.bool),
        },
        {
            "videos": torch.ones((2, 2, 2, 2, 3), dtype=torch.uint8),
            "task_descriptions": ["task b", "task c"],
            "last_run": torch.ones((2, 1), dtype=torch.bool),
        },
    ]

    merged = RoboRewardEmbodiedRewardWorker._merge_reward_input_batches(payloads)

    assert merged["videos"].shape == (3, 2, 2, 2, 3)
    assert merged["task_descriptions"] == ["task a", "task b", "task c"]
    assert RoboRewardEmbodiedRewardWorker._infer_reward_batch_size(merged) == 3


def test_roboreward_parse_and_map_scores():
    assert RoboRewardModel.parse_reward_score("ANSWER: 4") == 4
    assert RoboRewardModel.parse_reward_score("Final score 5") == 5

    scores = torch.tensor([1.0, 3.0, 5.0])
    normalized = RoboRewardModel.map_scores_to_rewards(scores, "normalized")
    centered = RoboRewardModel.map_scores_to_rewards(scores, "centered")

    assert torch.allclose(normalized, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(centered, torch.tensor([-1.0, 0.0, 1.0]))


def test_roboreward_invalid_score_fallback():
    model = RoboRewardModel.__new__(RoboRewardModel)
    model.invalid_score_policy = "fallback"
    model.invalid_score = 1
    model.warn_invalid_scores = False

    assert model._parse_reward_score_with_fallback("using 3D object tracking") == 1

    model.invalid_score_policy = "error"
    try:
        model._parse_reward_score_with_fallback("using 3D object tracking")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected malformed RoboReward output to raise")
