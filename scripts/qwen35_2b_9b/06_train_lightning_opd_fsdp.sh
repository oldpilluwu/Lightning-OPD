#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SFT_CHECKPOINT="$(find_latest_sft_checkpoint)"
LIGHTNING_OPD_DATA="${LIGHTNING_OPD_DATA:-${LIGHTNING_OPD_DIR}/$(basename "${ROLLOUT_DATA}" .parquet)-lightning-opd-precomputed.parquet}"

if [[ ! -f "${LIGHTNING_OPD_DATA}" ]]; then
    echo "Missing LIGHTNING_OPD_DATA: ${LIGHTNING_OPD_DATA}" >&2
    echo "Run scripts/qwen35_2b_9b/05_precompute_teacher_logprobs.sh first." >&2
    exit 1
fi

NUM_GPUS="${NUM_GPUS}" \
SFT_CHECKPOINT="${SFT_CHECKPOINT}" \
LIGHTNING_OPD_DATA="${LIGHTNING_OPD_DATA}" \
SAVE_DIR="${LIGHTNING_OPD_OUTPUT_DIR}" \
python configs/lightning_opd/qwen35-2b-lightning-opd-fsdp.py

echo "Lightning-OPD checkpoint dir: ${LIGHTNING_OPD_OUTPUT_DIR}"
