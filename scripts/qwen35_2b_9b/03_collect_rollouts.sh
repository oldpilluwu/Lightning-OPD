#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SFT_CHECKPOINT="$(find_latest_sft_checkpoint)"
VLLM_CHECKPOINT="${SFT_VLLM_CHECKPOINT:-${SFT_VLLM_DIR}/$(basename "${SFT_CHECKPOINT}")}"
RAW_DIR="${EXP_DIR}/rollouts/raw"
rm -rf "${RAW_DIR}"
mkdir -p "${RAW_DIR}" "$(dirname "${ROLLOUT_DATA}")"

python scripts/qwen35_2b_9b/prepare_vllm_checkpoint.py \
    --sft-checkpoint "${SFT_CHECKPOINT}" \
    --base-model "${STUDENT_BASE}" \
    --output-dir "${VLLM_CHECKPOINT}" \
    --force

SFT_CHECKPOINT="${VLLM_CHECKPOINT}" \
OPD_PROMPTS="${OPD_PROMPTS}" \
OUTPUT_DIR="${RAW_DIR}" \
NUM_GPUS="${NUM_GPUS}" \
TP_SIZE="${TP_SIZE}" \
bash scripts/collect_rollouts.sh \
    --num-samples "${OPD_NUM_SAMPLES}" \
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
    --output "${ROLLOUT_DATA}"

echo "SFT checkpoint: ${SFT_CHECKPOINT}"
echo "vLLM checkpoint: ${VLLM_CHECKPOINT}"
echo "Rollout parquet: ${ROLLOUT_DATA}"
