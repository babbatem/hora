#!/bin/bash
CACHE=$1
MESA_VK_DEVICE_SELECT=10de:2520 python train.py task=FrankaCubeSlide headless=False pipeline=gpu \
task.env.numEnvs=64 test=True \
train.algo=PPO \
train.ppo.priv_info=True train.ppo.proprio_adapt=False \
train.ppo.output_name=RMA_no_goal_noise/ \
checkpoint=outputs/RMA_no_goal_noise/stage1_nn/best.pth
