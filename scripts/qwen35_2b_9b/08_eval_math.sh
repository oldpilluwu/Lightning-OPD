#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EVAL_MODEL="${EVAL_MODEL:-${HF_EXPORT_DIR:-${CKPT_DIR}/lightning_opd_hf}}"
EVAL_INPUT="${EVAL_INPUT:-${OPD_PROMPTS}}"
EVAL_OUTPUT="${EVAL_OUTPUT:-logs/qwen35_2b_9b/eval_math.json}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-200}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-2048}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_TOP_P="${EVAL_TOP_P:-1.0}"

python scripts/qwen35_2b_9b/eval_math.py \
    --model "${EVAL_MODEL}" \
    --input "${EVAL_INPUT}" \
    --output "${EVAL_OUTPUT}" \
    --num-samples "${EVAL_NUM_SAMPLES}" \
    --max-tokens "${EVAL_MAX_TOKENS}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --temperature "${EVAL_TEMPERATURE}" \
    --top-p "${EVAL_TOP_P}"
