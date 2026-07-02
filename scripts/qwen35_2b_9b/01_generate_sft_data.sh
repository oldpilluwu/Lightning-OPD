#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RAW_DIR="${EXP_DIR}/sft_data/raw"
if [[ "${RESUME:-0}" != "1" ]]; then
    rm -rf "${RAW_DIR}"
fi
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
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
    --dtype "${VLLM_DTYPE}" \
    "$@"

python data_curation/merge.py \
    --input-dir "${RAW_DIR}" \
    --output "${SFT_ALL_DATA}"

python scripts/qwen35_2b_9b/split_sft_probe.py \
    --input "${SFT_ALL_DATA}" \
    --train-output "${SFT_DATA}" \
    --probe-output "${SFT_PROBE_DATA}" \
    --probe-size "${SFT_PROBE_SIZE}"

if [[ "${KEEP_RAW_SFT_SHARDS:-0}" != "1" ]]; then
    rm -rf "${RAW_DIR}"
fi

if [[ "${KEEP_SFT_ALL_DATA:-0}" != "1" ]]; then
    rm -f "${SFT_ALL_DATA}"
fi

if [[ -f "${SFT_ALL_DATA}" ]]; then
    echo "All SFT parquet: ${SFT_ALL_DATA}"
else
    echo "All SFT parquet removed after split. Set KEEP_SFT_ALL_DATA=1 to retain it."
fi
echo "SFT parquet: ${SFT_DATA}"
echo "SFT probe parquet: ${SFT_PROBE_DATA}"
