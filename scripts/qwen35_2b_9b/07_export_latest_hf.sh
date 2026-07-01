#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SFT_CHECKPOINT="$(find_latest_sft_checkpoint)"
LIGHTNING_OPD_OUTPUT_DIR="${LIGHTNING_OPD_OUTPUT_DIR:-${CKPT_DIR}/lightning_opd_fsdp}"
HF_EXPORT_DIR="${HF_EXPORT_DIR:-${CKPT_DIR}/lightning_opd_hf}"

TRACKER="${LIGHTNING_OPD_OUTPUT_DIR}/latest_checkpointed_iteration.txt"
if [[ ! -f "${TRACKER}" ]]; then
    echo "Missing FSDP tracker: ${TRACKER}" >&2
    exit 1
fi

STEP="$(cat "${TRACKER}")"
INPUT_DIR="${LIGHTNING_OPD_OUTPUT_DIR}/iter_$(printf "%07d" "${STEP}")"

python tools/convert_fsdp_to_hf.py \
    --origin-hf-dir "${SFT_CHECKPOINT}" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${HF_EXPORT_DIR}" \
    --force

echo "Exported HF model: ${HF_EXPORT_DIR}"
