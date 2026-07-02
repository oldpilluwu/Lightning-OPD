#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="${EXP_DIR:-data/qwen35_2b_9b}"
CKPT_DIR="${CKPT_DIR:-checkpoints/qwen35_2b_9b}"

STUDENT_BASE="${STUDENT_BASE:-Qwen/Qwen3.5-2B-Base}"
STUDENT_CHAT="${STUDENT_CHAT:-Qwen/Qwen3.5-2B}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-9B}"

NUM_GPUS="${NUM_GPUS:-1}"
TP_SIZE="${TP_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"

SFT_NUM_SAMPLES="${SFT_NUM_SAMPLES:-5000}"
OPD_NUM_SAMPLES="${OPD_NUM_SAMPLES:-2000}"

SFT_PROMPTS="${SFT_PROMPTS:-${EXP_DIR}/prompts/openthoughts3_${SFT_NUM_SAMPLES}.jsonl}"
OPD_PROMPTS="${OPD_PROMPTS:-${EXP_DIR}/prompts/dapo-math-17k/dapo-math-17k.jsonl}"
SFT_ALL_DATA="${SFT_ALL_DATA:-${EXP_DIR}/sft_data/all.parquet}"
SFT_DATA="${SFT_DATA:-${EXP_DIR}/sft_data/train.parquet}"
SFT_PROBE_DATA="${SFT_PROBE_DATA:-${EXP_DIR}/sft_data/probe.parquet}"
SFT_PROBE_PRECOMPUTED="${SFT_PROBE_PRECOMPUTED:-${EXP_DIR}/sft_data/probe_teacher_logprobs.parquet}"
SFT_PROBE_SIZE="${SFT_PROBE_SIZE:-512}"
ROLLOUT_DATA="${ROLLOUT_DATA:-${EXP_DIR}/rollouts/dapo-math-17k-qwen35-2b-sft-rollouts.parquet}"
LIGHTNING_OPD_DIR="${LIGHTNING_OPD_DIR:-${EXP_DIR}/lightning_opd}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${CKPT_DIR}/sft}"
LIGHTNING_OPD_OUTPUT_DIR="${LIGHTNING_OPD_OUTPUT_DIR:-${CKPT_DIR}/lightning_opd_fsdp}"

cd "${ROOT_DIR}"

mkdir -p "${EXP_DIR}" "${CKPT_DIR}"

find_latest_sft_checkpoint() {
    if [[ -n "${SFT_CHECKPOINT:-}" ]]; then
        echo "${SFT_CHECKPOINT}"
        return
    fi

    if compgen -G "${SFT_OUTPUT_DIR}/checkpoint-*" > /dev/null; then
        find "${SFT_OUTPUT_DIR}" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -n 1
        return
    fi

    echo "${SFT_OUTPUT_DIR}"
}
