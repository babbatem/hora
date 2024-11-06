#!/bin/bash
CACHE=$1
python train.py task=FrankaCubePush headless=False pipeline=gpu \
task.env.numEnvs=1 test=True \
train.algo=PPO \
train.ppo.priv_info=True \
train.ppo.output_name=FrankaCubePush/ \
checkpoint=outputs/FrankaCubePush/stage1_nn/best.pth
