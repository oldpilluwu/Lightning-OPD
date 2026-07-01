#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SFT_CHECKPOINT="$(find_latest_sft_checkpoint)"
TEACHER_URL="${TEACHER_URL:-http://127.0.0.1:13141/generate}"
PRECOMPUTE_CONCURRENCY="${PRECOMPUTE_CONCURRENCY:-16}"
PRECOMPUTE_MAX_RESPONSE_LEN="${PRECOMPUTE_MAX_RESPONSE_LEN:-2048}"
TEACHER_STARTED_BY_THIS_SCRIPT=0

cleanup_teacher() {
    if [[ "${TEACHER_STARTED_BY_THIS_SCRIPT}" == "1" && "${KEEP_TEACHER:-0}" != "1" ]]; then
        pkill -f "sglang.launch_server.*${TEACHER_MODEL}" || true
    fi
}
trap cleanup_teacher EXIT

if [[ "${START_TEACHER:-1}" == "1" ]]; then
    bash scripts/qwen35_2b_9b/04_serve_teacher.sh
    TEACHER_STARTED_BY_THIS_SCRIPT=1
fi

python data_curation/prepare_lightning_opd.py \
    --tokenizer-path "${SFT_CHECKPOINT}" \
    --input-parquet "${ROLLOUT_DATA}" \
    --output-dir "${LIGHTNING_OPD_DIR}" \
    --max-response-len "${PRECOMPUTE_MAX_RESPONSE_LEN}" \
    --compute-teacher-logprobs \
    --teacher-url "${TEACHER_URL}" \
    --concurrency "${PRECOMPUTE_CONCURRENCY}"

echo "Lightning-OPD data dir: ${LIGHTNING_OPD_DIR}"
