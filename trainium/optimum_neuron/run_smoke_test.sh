#!/usr/bin/env bash
# Prepare 1,000 examples and run two full-model Qwen3 optimizer steps on TP8.

set -euo pipefail

unset PYTHONPATH
export PYTHONNOUSERSITE=1

: "${SFT_DATA:?Set SFT_DATA to the generated Parquet file or shard directory}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Base}"
EXPECTED_ROWS="${EXPECTED_ROWS:-300000}"
SMOKE_SEQUENCE_LENGTH="${SMOKE_SEQUENCE_LENGTH:-1024}"
SMOKE_SOURCE_ROWS="${SMOKE_SOURCE_ROWS:-1000}"
SMOKE_ROOT="${SMOKE_ROOT:-${HOME}/lightning-opd-smoke}"
TOKENIZED_DATA="${TOKENIZED_DATA:-${SMOKE_ROOT}/packed-${SMOKE_SEQUENCE_LENGTH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SMOKE_ROOT}/run-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_FILE="${OUTPUT_DIR}.log"

mkdir -p "${SMOKE_ROOT}"
if [[ ! -f "${TOKENIZED_DATA}/reproduction_metadata.json" ]]; then
    python "${SCRIPT_DIR}/prepare_sft_dataset.py" \
        --dataset-path "${SFT_DATA}" \
        --output-dir "${TOKENIZED_DATA}" \
        --model-id "${MODEL_ID}" \
        --cutoff-len "${SMOKE_SEQUENCE_LENGTH}" \
        --expected-rows "${EXPECTED_ROWS}" \
        --max-samples "${SMOKE_SOURCE_ROWS}" \
        --preprocessing-num-workers 1 \
        --preprocessing-batch-size 1000
else
    echo "Reusing smoke dataset: ${TOKENIZED_DATA}"
fi

echo "Smoke output: ${OUTPUT_DIR}"
echo "Smoke log:    ${LOG_FILE}"

cd "${REPO_DIR}"
TOKENIZED_DATA="${TOKENIZED_DATA}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
MODEL_ID="${MODEL_ID}" \
SEQUENCE_LENGTH="${SMOKE_SEQUENCE_LENGTH}" \
SMOKE=1 \
bash "${SCRIPT_DIR}/run_sft_trn2_48xlarge.sh" \
    2>&1 | tee "${LOG_FILE}"

if [[ ! -f "${OUTPUT_DIR}/trainer_state.json" ]]; then
    echo "ERROR: training finished without trainer_state.json" >&2
    exit 1
fi

echo
echo "SMOKE_TEST_OK"
echo "Checkpoint directory: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
