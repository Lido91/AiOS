#!/bin/bash


CHECKPOINT=$1
INPUT_VIDEO=$2
OUTPUT_DIR=$3
NUM_PERSON=${4:-1}
THRESHOLD=${5:-0.3}
GPU_NUM=${6:-2}
ID_FILE=${7:-}  # Optional id.txt file path
export CUDA_VISIBLE_DEVICES=1,2

# Build the command with optional id_file parameter
CMD="torchrun \
    --nproc_per_node ${GPU_NUM} \
    main.py \
    -c config/aios_smplx_demo.py \
    --options batch_size=8 backbone=resnet50 num_person=${NUM_PERSON} threshold=${THRESHOLD} \
    --resume ${CHECKPOINT} \
    --eval \
    --inference \
    --inference_input ${INPUT_VIDEO} \
    --output_dir ${OUTPUT_DIR}"



# Add id_file parameter if providedí
if [ -n "${ID_FILE}" ]; then
    CMD="${CMD} --id_file ${ID_FILE}"
fi

# Execute the command
eval ${CMD}
    # --debug
    # --to_vid 



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
