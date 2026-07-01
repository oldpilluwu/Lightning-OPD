#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SFT_CHECKPOINT="$(find_latest_sft_checkpoint)"
RAW_DIR="${EXP_DIR}/rollouts/raw"
rm -rf "${RAW_DIR}"
mkdir -p "${RAW_DIR}" "$(dirname "${ROLLOUT_DATA}")"

SFT_CHECKPOINT="${SFT_CHECKPOINT}" \
OPD_PROMPTS="${OPD_PROMPTS}" \
OUTPUT_DIR="${RAW_DIR}" \
NUM_GPUS="${NUM_GPUS}" \
TP_SIZE="${TP_SIZE}" \
bash scripts/collect_rollouts.sh \
    --num-samples "${OPD_NUM_SAMPLES}" \
    --max-tokens "${MAX_TOKENS}" \
    --batch-size "${BATCH_SIZE}" \
    "$@"

python data_curation/merge.py \
    --input-dir "${RAW_DIR}" \
    --output "${ROLLOUT_DATA}"

echo "SFT checkpoint: ${SFT_CHECKPOINT}"
echo "Rollout parquet: ${ROLLOUT_DATA}"
