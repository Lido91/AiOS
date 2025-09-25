#!/bin/bash


CHECKPOINT=$1
INPUT_VIDEO=$2
OUTPUT_DIR=$3
NUM_PERSON=${4:-1}
THRESHOLD=${5:-0.3}
GPU_NUM=${6:-8}
# export CUDA_VISIBLE_DEVICES=${GPU_ID}
python -m torch.distributed.launch \
    --nproc_per_node ${GPU_NUM} \
    main.py \
    -c "config/aios_smplx_demo.py" \
    --options batch_size=8 backbone="resnet50" num_person=${NUM_PERSON} threshold=${THRESHOLD} \
    --resume ${CHECKPOINT} \
    --eval \
    --inference \
    --to_vid \
    --inference_input ${INPUT_VIDEO} \
    --output_dir demo/${OUTPUT_DIR} \
    # --debug



#!/usr/bin/env bash
# run_single_ddp.sh
# set -euo pipefail

# CHECKPOINT=${1:?}
# INPUT_VIDEO=${2:?}
# OUTPUT_DIR=${3:?}
# NUM_PERSON=${4:-1}
# THRESHOLD=${5:-0.3}
# BATCH_SIZE=${BATCH_SIZE:-8}
# GPU_ID=${GPU_ID:-2}

# export CUDA_VISIBLE_DEVICES=${GPU_ID}

# # Prefer torchrun (newer) over deprecated torch.distributed.launch
# torchrun --nproc_per_node=1 main.py \
#   -c "config/aios_smplx_demo.py" \
#   --options batch_size=${BATCH_SIZE} backbone="resnet50" num_person=${NUM_PERSON} threshold=${THRESHOLD} \
#   --resume "${CHECKPOINT}" \
#   --eval \
#   --inference \
#   --to_vid \
#   --inference_input "${INPUT_VIDEO}" \
#   --output_dir "demo/${OUTPUT_DIR}"

