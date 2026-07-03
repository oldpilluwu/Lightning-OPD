#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_BIN="${CONDA_BIN:-conda}"
if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
    if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
        CONDA_BIN="${HOME}/miniconda3/bin/conda"
    else
        echo "conda not found. Set CONDA_BIN=/path/to/conda." >&2
        exit 1
    fi
fi

TRAIN_ENV="${TRAIN_ENV:-qwen35-train}"
CURATION_ENV="${CURATION_ENV:-qwen35-curation}"
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
RUN_ID="${RUN_ID:-continue_after_opd_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/qwen35_2b_9b/${RUN_ID}}"
mkdir -p "${LOG_DIR}"

source scripts/qwen35_2b_9b/common.sh

export LOG_DIR RUN_ID
export LIGHTNING_OPD_OUTPUT_DIR="${LIGHTNING_OPD_OUTPUT_DIR:-${CKPT_DIR}/lightning_opd_fsdp}"
export HF_EXPORT_DIR="${HF_EXPORT_DIR:-${CKPT_DIR}/lightning_opd_hf}"

ray_job_id="$(
    "${CONDA_BIN}" run --no-capture-output -n "${TRAIN_ENV}" python - "${RAY_ADDRESS}" <<'PY'
import sys

from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient(sys.argv[1])
jobs = client.list_jobs()
items = list(jobs.items()) if isinstance(jobs, dict) else [
    (getattr(job, "submission_id", None) or getattr(job, "job_id", None), job)
    for job in jobs
]

def field(job, name, default=None):
    return job.get(name, default) if isinstance(job, dict) else getattr(job, name, default)

def status(job):
    return str(field(job, "status", "")).split(".")[-1].upper()

def ts(item):
    return field(item[1], "start_time", None) or 0

active = [item for item in items if status(item[1]) in {"PENDING", "RUNNING"}]
pool = active or items
if not pool:
    raise SystemExit("No Ray jobs found.")

job_id, _ = max(pool, key=ts)
print(job_id)
PY
)"

echo "Watching Ray job: ${ray_job_id}"
echo "Logs will go to: ${LOG_DIR}"

while true; do
    status="$(
        "${CONDA_BIN}" run --no-capture-output -n "${TRAIN_ENV}" python - "${RAY_ADDRESS}" "${ray_job_id}" <<'PY'
import sys

from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient(sys.argv[1])
info = client.get_job_info(sys.argv[2])
print(str(info.status).split(".")[-1].upper())
PY
    )"
    echo "$(date '+%F %T') Ray job ${ray_job_id}: ${status}"
    case "${status}" in
        SUCCEEDED)
            break
            ;;
        FAILED|STOPPED)
            echo "Ray job did not succeed. Tail follows:" >&2
            "${CONDA_BIN}" run --no-capture-output -n "${TRAIN_ENV}" \
                ray job logs --address="${RAY_ADDRESS}" "${ray_job_id}" | tail -n 300 >&2
            exit 1
            ;;
    esac
    sleep "${POLL_SECONDS:-60}"
done

"${CONDA_BIN}" run --no-capture-output -n "${TRAIN_ENV}" \
    ray job logs --address="${RAY_ADDRESS}" "${ray_job_id}" > "${LOG_DIR}/06_train_lightning_opd_fsdp.log" || true

run_stage() {
    local env_name="$1"
    local stage_name="$2"
    shift 2
    local log_path="${LOG_DIR}/${stage_name}.log"

    echo "=== ${stage_name} | env=${env_name} | log=${log_path} ==="
    "${CONDA_BIN}" run --no-capture-output -n "${env_name}" bash -lc "$*" 2>&1 | tee "${log_path}"
}

run_stage "${TRAIN_ENV}" "inspect_drift" \
    "python scripts/qwen35_2b_9b/inspect_drift.py --log-dir '${LOG_DIR}' --output-dir '${LOG_DIR}/drift'"
run_stage "${TRAIN_ENV}" "07_export_latest_hf" \
    "bash scripts/qwen35_2b_9b/07_export_latest_hf.sh"
run_stage "${CURATION_ENV}" "08_eval_sft" \
    "EVAL_MODEL=\"\$(bash scripts/qwen35_2b_9b/print_latest_sft_checkpoint.sh)\" EVAL_OUTPUT='${LOG_DIR}/eval_sft.json' bash scripts/qwen35_2b_9b/08_eval_math.sh"
run_stage "${CURATION_ENV}" "08_eval_final" \
    "EVAL_MODEL='${HF_EXPORT_DIR}' EVAL_OUTPUT='${LOG_DIR}/eval_final.json' bash scripts/qwen35_2b_9b/08_eval_math.sh"

echo "Continuation done."
echo "Logs: ${LOG_DIR}"
echo "Drift summary: ${LOG_DIR}/drift/summary.txt"
echo "Eval baseline: ${LOG_DIR}/eval_sft.json"
echo "Eval final: ${LOG_DIR}/eval_final.json"
