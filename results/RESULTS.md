# RLinf Experimental Results

Running record of OpenVLA-OFT GRPO experiments on `PutOnPlateInScene25Main-v3`.

## Training Runs

| Run | Reward source | Epochs completed | Training envs | Notes | Log directory |
| --- | --- | ---: | ---: | --- | --- |
| RoboReward GRPO | RoboReward-8B terminal video reward | 500 | 32 | OpenVLA-OFT actor trained with GRPO, group size 8, terminal RoboReward reward, no env-reward mixing. Checkpoints 100-400 are in the first log directory; epoch 500 is in the resumed/final log directory. | `logs/20260517-22:23:06-maniskill_grpo_openvlaoft_roboreward`, `logs/20260519-10:14:00-maniskill_grpo_openvlaoft_roboreward` |
| Env-reward GRPO | ManiSkill environment reward | 500 | 32 | OpenVLA-OFT actor trained with GRPO, group size 8, environment reward. Training resumed across preemptions/timeouts; final run completed epoch 500. | `logs/20260521-08:49:25-maniskill_grpo_openvlaoft`, `logs/20260521-16:03:08-maniskill_grpo_openvlaoft`, `logs/20260521-21:30:23-maniskill_grpo_openvlaoft`, `logs/20260522-20:51:10-maniskill_grpo_openvlaoft` |

## Completed Evaluations

All rows below are standard eval rollouts scored with both environment metrics and RoboReward post-hoc scoring. The newer checkpoint evals use 64 parallel envs, 8 eval rollout epochs, `auto_reset=False`, and 512 total rollouts. `roboreward_avg` is the mean per-rollout terminal RoboReward sum after normalized score mapping `(raw_score - 1) / 4`.

| Model | Checkpoint epoch | Success once | RoboReward avg | Rollouts | Eval log directory |
| --- | ---: | ---: | ---: | ---: | --- |
| Env-reward GRPO | 100 | 0.2773 | 0.0537 | 512 | `logs/eval/envreward-gs100-eval-20260524-202400` |
| Env-reward GRPO | 200 | 0.3301 | 0.0703 | 512 | `logs/eval/envreward-gs200-eval-20260524-201707` |
| Env-reward GRPO | 300 | 0.5449 | 0.0596 | 512 | `logs/eval/envreward-gs300-eval-20260525-114430` |
| Env-reward GRPO | 400 | 0.6426 | 0.0400 | 512 | `logs/eval/envreward-gs400-eval-20260525-114430` |
| Env-reward GRPO | 500 | 0.6895 | 0.0264 | 512 | `logs/eval/20260524-160354-envreward-gs500-roboreward_rollout` |
| RoboReward GRPO | 100 | 0.3652 | 0.0703 | 512 | `logs/eval/roboreward-gs100-eval-20260524-201707` |
| RoboReward GRPO | 200 | 0.5566 | 0.0498 | 512 | `logs/eval/roboreward-gs200-eval-20260524-201707` |
| RoboReward GRPO | 300 | 0.5664 | 0.0420 | 512 | `logs/eval/roboreward-gs300-eval-20260525-114430` |
| RoboReward GRPO | 400 | 0.6387 | 0.0435 | 512 | `logs/eval/roboreward-gs400-eval-20260524-201707` |
| RoboReward GRPO | 500 | 0.6797 | 0.3057 | 512 | `logs/eval/20260521-085043-roboreward-gs500-roboreward_rollout` |
