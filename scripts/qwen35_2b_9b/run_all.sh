#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_BIN="${CONDA_BIN:-conda}"
CURATION_ENV="${CURATION_ENV:-qwen35-curation}"
SFT_ENV="${SFT_ENV:-qwen35-sft}"
TRAIN_ENV="${TRAIN_ENV:-qwen35-train}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/qwen35_2b_9b/${RUN_ID}}"
mkdir -p "${LOG_DIR}"

export RUN_ID LOG_DIR
export USE_TENSORBOARD="${USE_TENSORBOARD:-1}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${LOG_DIR}/tensorboard}"
export LIGHTNING_OPD_OUTPUT_DIR="${LIGHTNING_OPD_OUTPUT_DIR:-checkpoints/qwen35_2b_9b/lightning_opd_fsdp_${RUN_ID}}"
export HF_EXPORT_DIR="${HF_EXPORT_DIR:-checkpoints/qwen35_2b_9b/lightning_opd_hf_${RUN_ID}}"

run_stage() {
    local env_name="$1"
    local stage_name="$2"
    shift 2
    local log_path="${LOG_DIR}/${stage_name}.log"

    echo "=== ${stage_name} | env=${env_name} | log=${log_path} ==="
    "${CONDA_BIN}" run -n "${env_name}" bash -lc "$*" 2>&1 | tee "${log_path}"
}

{
    echo "run_id=${RUN_ID}"
    echo "log_dir=${LOG_DIR}"
    echo "curation_env=${CURATION_ENV}"
    echo "sft_env=${SFT_ENV}"
    echo "train_env=${TRAIN_ENV}"
    echo "num_gpus=${NUM_GPUS:-1}"
    echo "sft_num_samples=${SFT_NUM_SAMPLES:-5000}"
    echo "opd_num_samples=${OPD_NUM_SAMPLES:-2000}"
    echo "sft_max_steps=${SFT_MAX_STEPS:-500}"
    echo "num_rollout=${NUM_ROLLOUT:-200}"
    echo "lightning_opd_output_dir=${LIGHTNING_OPD_OUTPUT_DIR}"
    echo "hf_export_dir=${HF_EXPORT_DIR}"
} | tee "${LOG_DIR}/run.env"

run_stage "${CURATION_ENV}" "00_prepare_prompts" "bash scripts/qwen35_2b_9b/00_prepare_prompts.sh"
run_stage "${CURATION_ENV}" "01_generate_sft_data" "bash scripts/qwen35_2b_9b/01_generate_sft_data.sh"
run_stage "${SFT_ENV}" "02_run_sft" "bash scripts/qwen35_2b_9b/02_run_sft.sh"
run_stage "${CURATION_ENV}" "03_collect_rollouts" "bash scripts/qwen35_2b_9b/03_collect_rollouts.sh"
run_stage "${TRAIN_ENV}" "05_precompute_teacher_logprobs" "bash scripts/qwen35_2b_9b/05_precompute_teacher_logprobs.sh"
run_stage "${TRAIN_ENV}" "06_train_lightning_opd_fsdp" "bash scripts/qwen35_2b_9b/06_train_lightning_opd_fsdp.sh"
run_stage "${TRAIN_ENV}" "inspect_drift" "python scripts/qwen35_2b_9b/inspect_drift.py --log-dir '${LOG_DIR}' --output-dir '${LOG_DIR}/drift'"
run_stage "${TRAIN_ENV}" "07_export_latest_hf" "bash scripts/qwen35_2b_9b/07_export_latest_hf.sh"
run_stage "${CURATION_ENV}" "08_eval_sft" "EVAL_MODEL=\"\$(bash scripts/qwen35_2b_9b/print_latest_sft_checkpoint.sh)\" EVAL_OUTPUT='${LOG_DIR}/eval_sft.json' bash scripts/qwen35_2b_9b/08_eval_math.sh"
run_stage "${CURATION_ENV}" "08_eval_final" "EVAL_MODEL='${HF_EXPORT_DIR}' EVAL_OUTPUT='${LOG_DIR}/eval_final.json' bash scripts/qwen35_2b_9b/08_eval_math.sh"

echo "All done."
echo "Logs: ${LOG_DIR}"
echo "Drift summary: ${LOG_DIR}/drift/summary.txt"
echo "Eval baseline: ${LOG_DIR}/eval_sft.json"
echo "Eval final: ${LOG_DIR}/eval_final.json"
