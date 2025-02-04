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
train.algo=ProprioAdapt \
train.ppo.priv_info=True train.ppo.proprio_adapt=True \
train.ppo.output_name=RMA_no_goal_noise/"${CACHE}" \
checkpoint=outputs/RMA_no_goal_noise/"${CACHE}"/stage1_nn/best.pth \
${EXTRA_ARGS}