#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_BIN="${CONDA_BIN:-conda}"
if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
    if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
        CONDA_BIN="${HOME}/miniconda3/bin/conda"
    else
        echo "conda not found. Run scripts/qwen35_2b_9b/setup_remote.sh first, or set CONDA_BIN=/path/to/conda." >&2
        exit 1
    fi
fi

CURATION_ENV="${CURATION_ENV:-qwen35-curation}"
TRAIN_ENV="${TRAIN_ENV:-qwen35-train}"

RUN_ID="${RUN_ID:-resume_rollout_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/qwen35_2b_9b/${RUN_ID}}"
mkdir -p "${LOG_DIR}"

export RUN_ID LOG_DIR
export USE_TENSORBOARD="${USE_TENSORBOARD:-1}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${LOG_DIR}/tensorboard}"
export LIGHTNING_OPD_OUTPUT_DIR="${LIGHTNING_OPD_OUTPUT_DIR:-checkpoints/qwen35_2b_9b/lightning_opd_fsdp_${RUN_ID}}"
export HF_EXPORT_DIR="${HF_EXPORT_DIR:-checkpoints/qwen35_2b_9b/lightning_opd_hf_${RUN_ID}}"

source scripts/qwen35_2b_9b/common.sh

run_stage() {
    local env_name="$1"
    local stage_name="$2"
    shift 2
    local log_path="${LOG_DIR}/${stage_name}.log"

    echo "=== ${stage_name} | env=${env_name} | log=${log_path} ==="
    "${CONDA_BIN}" run --no-capture-output -n "${env_name}" bash -lc "$*" 2>&1 | tee "${log_path}"
}

if [[ -z "${SFT_CHECKPOINT:-}" ]]; then
    export SFT_CHECKPOINT="$(find_latest_sft_checkpoint)"
fi

if [[ ! -d "${SFT_CHECKPOINT}" ]]; then
    echo "Missing SFT checkpoint directory: ${SFT_CHECKPOINT}" >&2
    exit 1
fi

if [[ ! -f "${OPD_PROMPTS}" ]]; then
    echo "Missing OPD prompts: ${OPD_PROMPTS}" >&2
    exit 1
fi

{
    echo "run_id=${RUN_ID}"
    echo "log_dir=${LOG_DIR}"
    echo "resume_from=rollout"
    echo "curation_env=${CURATION_ENV}"
    echo "train_env=${TRAIN_ENV}"
    echo "sft_checkpoint=${SFT_CHECKPOINT}"
    echo "opd_prompts=${OPD_PROMPTS}"
    echo "rollout_data=${ROLLOUT_DATA}"
    echo "lightning_opd_output_dir=${LIGHTNING_OPD_OUTPUT_DIR}"
    echo "hf_export_dir=${HF_EXPORT_DIR}"
} | tee "${LOG_DIR}/run.env"

run_stage "${CURATION_ENV}" "03_collect_rollouts" "bash scripts/qwen35_2b_9b/03_collect_rollouts.sh"
run_stage "${TRAIN_ENV}" "05_precompute_teacher_logprobs" "bash scripts/qwen35_2b_9b/05_precompute_teacher_logprobs.sh"
run_stage "${TRAIN_ENV}" "06_train_lightning_opd_fsdp" "bash scripts/qwen35_2b_9b/06_train_lightning_opd_fsdp.sh"
run_stage "${TRAIN_ENV}" "inspect_drift" "python scripts/qwen35_2b_9b/inspect_drift.py --log-dir '${LOG_DIR}' --output-dir '${LOG_DIR}/drift'"
run_stage "${TRAIN_ENV}" "07_export_latest_hf" "bash scripts/qwen35_2b_9b/07_export_latest_hf.sh"
run_stage "${CURATION_ENV}" "08_eval_sft" "EVAL_MODEL='${SFT_CHECKPOINT}' EVAL_OUTPUT='${LOG_DIR}/eval_sft.json' bash scripts/qwen35_2b_9b/08_eval_math.sh"
run_stage "${CURATION_ENV}" "08_eval_final" "EVAL_MODEL='${HF_EXPORT_DIR}' EVAL_OUTPUT='${LOG_DIR}/eval_final.json' bash scripts/qwen35_2b_9b/08_eval_math.sh"

echo "Resume-from-rollout pipeline done."
echo "Logs: ${LOG_DIR}"
echo "Drift summary: ${LOG_DIR}/drift/summary.txt"
echo "Eval baseline: ${LOG_DIR}/eval_sft.json"
echo "Eval final: ${LOG_DIR}/eval_final.json"
