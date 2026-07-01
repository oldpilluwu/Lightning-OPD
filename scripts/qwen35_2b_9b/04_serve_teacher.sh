#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_HOST="${TEACHER_HOST:-127.0.0.1}"
TEACHER_CONTEXT_LEN="${TEACHER_CONTEXT_LEN:-8192}"
TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION:-0.75}"
SGLANG_TP_FLAG="${SGLANG_TP_FLAG:---tp}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"

LOG_FILE="/tmp/sglang_qwen35_9b_$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6).log"

python3 -m sglang.launch_server \
    --model-path "${TEACHER_MODEL}" \
    --host "${TEACHER_HOST}" \
    --port "${TEACHER_PORT}" \
    "${SGLANG_TP_FLAG}" "${TP_SIZE}" \
    --chunked-prefill-size 4096 \
    --mem-fraction-static "${TEACHER_MEM_FRACTION}" \
    --context-length "${TEACHER_CONTEXT_LEN}" \
    --reasoning-parser qwen3 \
    ${SGLANG_EXTRA_ARGS} \
    > "${LOG_FILE}" 2>&1 &

until curl -sf "http://${TEACHER_HOST}:${TEACHER_PORT}/health_generate" > /dev/null; do
    echo "Waiting for Qwen3.5 teacher server..."
    tail -n 20 "${LOG_FILE}"
    sleep 5
done

echo "Teacher ready: http://${TEACHER_HOST}:${TEACHER_PORT}/generate"
echo "Log file: ${LOG_FILE}"
