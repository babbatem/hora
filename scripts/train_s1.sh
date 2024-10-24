#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=FrankaCubePush headless=True seed=${SEED} \
train.algo=PPO \
train.ppo.priv_info=True train.ppo.proprio_adapt=False \
train.ppo.output_name=FrankaCubePush/"${CACHE}" \
${EXTRA_ARGS}