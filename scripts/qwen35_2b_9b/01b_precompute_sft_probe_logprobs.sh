#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

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

python scripts/qwen35_2b_9b/prepare_sft_probe_logprobs.py \
    --tokenizer-path "${STUDENT_BASE}" \
    --input-parquet "${SFT_PROBE_DATA}" \
    --output-parquet "${SFT_PROBE_PRECOMPUTED}" \
    --teacher-url "${TEACHER_URL}" \
    --concurrency "${PRECOMPUTE_CONCURRENCY}" \
    --max-response-len "${PRECOMPUTE_MAX_RESPONSE_LEN}"

echo "SFT probe with teacher logprobs: ${SFT_PROBE_PRECOMPUTED}"
