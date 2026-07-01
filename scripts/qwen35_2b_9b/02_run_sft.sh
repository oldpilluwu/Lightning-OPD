#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

NUM_NODES="${NUM_NODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-500}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-50}"
SFT_SAVE_TOTAL_LIMIT="${SFT_SAVE_TOTAL_LIMIT:-3}"
SFT_CUTOFF_LEN="${SFT_CUTOFF_LEN:-8192}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-8}"

mkdir -p "${SFT_OUTPUT_DIR}"

torchrun \
    --nnodes "${NUM_NODES}" \
    --nproc_per_node="${NUM_GPUS}" \
    --rdzv_id "${RANDOM}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "${MASTER_ADDR}:29500" \
    -m llamafactory.cli.train \
    configs/sft/qwen35-2b-base-open-thoughts3-qwen35-9b.yaml \
    "dataset_dir=configs/sft" \
    "model_name_or_path=${STUDENT_BASE}" \
    "output_dir=${SFT_OUTPUT_DIR}" \
    "max_steps=${SFT_MAX_STEPS}" \
    "save_steps=${SFT_SAVE_STEPS}" \
    "save_total_limit=${SFT_SAVE_TOTAL_LIMIT}" \
    "cutoff_len=${SFT_CUTOFF_LEN}" \
    "per_device_train_batch_size=${SFT_BATCH_SIZE}" \
    "gradient_accumulation_steps=${SFT_GRAD_ACCUM}"

echo "SFT output: ${SFT_OUTPUT_DIR}"
