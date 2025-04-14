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
train.ppo.output_name=3_fr_0_11__0_11_lr_2e-4_clip_0_2_entropy_0_priv_False/"${CACHE}" \
checkpoint=outputs/2_fr_0_05_0_1_lr_2e-4_clip_0_2_entropy_0_priv_False/"${CACHE}"/stage2_nn/best.pth \
${EXTRA_ARGS}