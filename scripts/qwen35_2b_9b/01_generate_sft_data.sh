#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RAW_DIR="${EXP_DIR}/sft_data/raw"
rm -rf "${RAW_DIR}"
mkdir -p "${RAW_DIR}" "$(dirname "${SFT_DATA}")"

TEACHER_MODEL="${TEACHER_MODEL}" \
SFT_PROMPTS="${SFT_PROMPTS}" \
OUTPUT_DIR="${RAW_DIR}" \
NUM_GPUS="${NUM_GPUS}" \
TP_SIZE="${TP_SIZE}" \
bash scripts/generate_sft_data.sh \
    --num-samples "${SFT_NUM_SAMPLES}" \
    --max-tokens "${MAX_TOKENS}" \
    --batch-size "${BATCH_SIZE}" \
    "$@"

python data_curation/merge.py \
    --input-dir "${RAW_DIR}" \
    --output "${SFT_DATA}"

echo "SFT parquet: ${SFT_DATA}"
