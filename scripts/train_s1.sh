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
python train.py task=FrankaCubeSlide headless=True seed=${SEED} \
train.algo=PPO \
train.ppo.priv_info=True train.ppo.proprio_adapt=False \
train.ppo.output_name=1_fr_0_05_0_075_256bins_lr_2e-4_clip_0_2_entropy_0_0_priv_True/"${CACHE}" \
${EXTRA_ARGS}