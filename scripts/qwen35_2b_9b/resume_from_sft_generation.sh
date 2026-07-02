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
SFT_ENV="${SFT_ENV:-qwen35-sft}"
TRAIN_ENV="${TRAIN_ENV:-qwen35-train}"

RUN_ID="${RUN_ID:-resume_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/qwen35_2b_9b/${RUN_ID}}"
mkdir -p "${LOG_DIR}"

export RUN_ID LOG_DIR
export RESUME=1
export BATCH_SIZE="${BATCH_SIZE:-16}"
export MAX_TOKENS="${MAX_TOKENS:-2048}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
export VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
export USE_TENSORBOARD="${USE_TENSORBOARD:-1}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${LOG_DIR}/tensorboard}"
export LIGHTNING_OPD_OUTPUT_DIR="${LIGHTNING_OPD_OUTPUT_DIR:-checkpoints/qwen35_2b_9b/lightning_opd_fsdp_${RUN_ID}}"
export HF_EXPORT_DIR="${HF_EXPORT_DIR:-checkpoints/qwen35_2b_9b/lightning_opd_hf_${RUN_ID}}"
export SFT_PROBE_METRICS_DIR="${SFT_PROBE_METRICS_DIR:-${LOG_DIR}/sft_probe_metrics}"

source scripts/qwen35_2b_9b/common.sh

run_stage() {
    local env_name="$1"
    local stage_name="$2"
    shift 2
    local log_path="${LOG_DIR}/${stage_name}.log"

    echo "=== ${stage_name} | env=${env_name} | log=${log_path} ==="
    "${CONDA_BIN}" run --no-capture-output -n "${env_name}" bash -lc "$*" 2>&1 | tee "${log_path}"
}

run_sft_with_monitor() {
    local sft_log="${LOG_DIR}/02_run_sft.log"
    local monitor_log="${LOG_DIR}/02_monitor_sft_saturation.log"

    echo "=== 02_monitor_sft_saturation | env=${SFT_ENV} | log=${monitor_log} ==="
    "${CONDA_BIN}" run --no-capture-output -n "${SFT_ENV}" bash -lc "bash scripts/qwen35_2b_9b/02_monitor_sft_saturation.sh" \
        > >(tee "${monitor_log}") 2>&1 &
    local monitor_pid=$!

    echo "=== 02_run_sft | env=${SFT_ENV} | log=${sft_log} ==="
    set +e
    "${CONDA_BIN}" run --no-capture-output -n "${SFT_ENV}" bash -lc "bash scripts/qwen35_2b_9b/02_run_sft.sh" \
        2>&1 | tee "${sft_log}"
    local sft_status=${PIPESTATUS[0]}
    set -e

    kill "${monitor_pid}" >/dev/null 2>&1 || true
    pkill -f "monitor_sft_saturation.py.*${SFT_OUTPUT_DIR}" >/dev/null 2>&1 || true
    wait "${monitor_pid}" >/dev/null 2>&1 || true

    echo "=== 02_monitor_sft_saturation_final | env=${SFT_ENV} | log=${monitor_log} ==="
    "${CONDA_BIN}" run --no-capture-output -n "${SFT_ENV}" bash -lc "python scripts/qwen35_2b_9b/monitor_sft_saturation.py --checkpoint-dir '${SFT_OUTPUT_DIR}' --probe-parquet '${SFT_PROBE_PRECOMPUTED}' --output-dir '${SFT_PROBE_METRICS_DIR}' --keep-latest '${SFT_PROBE_KEEP_LATEST:-1}' --plateau-threshold '${SFT_PROBE_PLATEAU_THRESHOLD:-0.01}' --plateau-patience '${SFT_PROBE_PLATEAU_PATIENCE:-3}'" \
        2>&1 | tee -a "${monitor_log}"

    return "${sft_status}"
}

{
    echo "run_id=${RUN_ID}"
    echo "log_dir=${LOG_DIR}"
    echo "resume_from=sft_generation"
    echo "curation_env=${CURATION_ENV}"
    echo "sft_env=${SFT_ENV}"
    echo "train_env=${TRAIN_ENV}"
    echo "raw_dir=${EXP_DIR}/sft_data/raw"
    echo "resume_checkpoint_dir=checkpoints"
    echo "batch_size=${BATCH_SIZE}"
    echo "max_tokens=${MAX_TOKENS}"
    echo "vllm_gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION}"
    echo "vllm_max_model_len=${VLLM_MAX_MODEL_LEN}"
    echo "vllm_max_num_seqs=${VLLM_MAX_NUM_SEQS}"
    echo "vllm_max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS}"
    echo "sft_probe_metrics_dir=${SFT_PROBE_METRICS_DIR}"
    echo "lightning_opd_output_dir=${LIGHTNING_OPD_OUTPUT_DIR}"
    echo "hf_export_dir=${HF_EXPORT_DIR}"
} | tee "${LOG_DIR}/run.env"

if [[ ! -d "${EXP_DIR}/sft_data/raw" ]]; then
    echo "Missing raw SFT output dir: ${EXP_DIR}/sft_data/raw" >&2
    echo "This resume script expects stage 01 to have started already." >&2
    exit 1
fi

if ! compgen -G "checkpoints/rank*.pkl" > /dev/null; then
    echo "Warning: no checkpoints/rank*.pkl found. Stage 01 may restart from the beginning of its raw dir."
fi

run_stage "${CURATION_ENV}" "01_generate_sft_data_resume" "bash scripts/qwen35_2b_9b/01_generate_sft_data.sh"
run_stage "${TRAIN_ENV}" "01b_precompute_sft_probe_logprobs" "bash scripts/qwen35_2b_9b/01b_precompute_sft_probe_logprobs.sh"
run_sft_with_monitor

if [[ -f "${SFT_PROBE_METRICS_DIR}/selected_checkpoint.txt" ]]; then
    export SFT_CHECKPOINT="$(cat "${SFT_PROBE_METRICS_DIR}/selected_checkpoint.txt")"
    echo "Selected SFT checkpoint for OPD: ${SFT_CHECKPOINT}" | tee -a "${LOG_DIR}/run.env"
fi

run_stage "${CURATION_ENV}" "03_collect_rollouts" "bash scripts/qwen35_2b_9b/03_collect_rollouts.sh"
run_stage "${TRAIN_ENV}" "05_precompute_teacher_logprobs" "bash scripts/qwen35_2b_9b/05_precompute_teacher_logprobs.sh"
run_stage "${TRAIN_ENV}" "06_train_lightning_opd_fsdp" "bash scripts/qwen35_2b_9b/06_train_lightning_opd_fsdp.sh"
run_stage "${TRAIN_ENV}" "inspect_drift" "python scripts/qwen35_2b_9b/inspect_drift.py --log-dir '${LOG_DIR}' --output-dir '${LOG_DIR}/drift'"
run_stage "${TRAIN_ENV}" "07_export_latest_hf" "bash scripts/qwen35_2b_9b/07_export_latest_hf.sh"
run_stage "${CURATION_ENV}" "08_eval_sft" "EVAL_MODEL=\"\$(bash scripts/qwen35_2b_9b/print_latest_sft_checkpoint.sh)\" EVAL_OUTPUT='${LOG_DIR}/eval_sft.json' bash scripts/qwen35_2b_9b/08_eval_math.sh"
run_stage "${CURATION_ENV}" "08_eval_final" "EVAL_MODEL='${HF_EXPORT_DIR}' EVAL_OUTPUT='${LOG_DIR}/eval_final.json' bash scripts/qwen35_2b_9b/08_eval_math.sh"

echo "Resume pipeline done."
echo "Logs: ${LOG_DIR}"
