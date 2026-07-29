#!/usr/bin/env bash
# Prepare a small exact dataset and execute two direct full-model steps.

set -euo pipefail

unset PYTHONPATH
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PREP_DIR="${PREP_DIR:-$(cd -- "${REPO_DIR}/.." && pwd)/lightning-opd-prep}"
SFT_DATA="${SFT_DATA:-${PREP_DIR}/sft_data/openthoughts3_300000_qwen3-8b.parquet}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Base}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-${PREP_DIR}/model_checkpoints/qwen3-4b-base-tp8}"
SMOKE_SEQUENCE_LENGTH="${SMOKE_SEQUENCE_LENGTH:-2048}"
SMOKE_SOURCE_ROWS="${SMOKE_SOURCE_ROWS:-1000}"
SMOKE_ROOT="${SMOKE_ROOT:-${HOME}/lightning-opd-pytorch29-smoke}"
TOKENIZED_DATA="${TOKENIZED_DATA:-${SMOKE_ROOT}/packed-rows${SMOKE_SOURCE_ROWS}-seq${SMOKE_SEQUENCE_LENGTH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SMOKE_ROOT}/run-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_FILE="${OUTPUT_DIR}.log"

if [[ ! -f "${SFT_DATA}" ]]; then
    echo "ERROR: SFT dataset not found: ${SFT_DATA}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_CHECKPOINT}/pretrained_weight/model/dp_rank_00_tp_rank_00_pp_rank_00.pt" ]]; then
    echo "Preparing direct TP=8 Qwen3 checkpoint..."
    MODEL_ID="${MODEL_ID}" \
    MODEL_CHECKPOINT="${MODEL_CHECKPOINT}" \
    PREP_DIR="${PREP_DIR}" \
        bash "${SCRIPT_DIR}/prepare_model_checkpoint.sh"
fi

mkdir -p "${SMOKE_ROOT}"
python "${SCRIPT_DIR}/prepare_sft_dataset.py" \
    --dataset-path "${SFT_DATA}" \
    --output-dir "${TOKENIZED_DATA}" \
    --model-id "${MODEL_ID}" \
    --cutoff-len "${SMOKE_SEQUENCE_LENGTH}" \
    --expected-rows 300000 \
    --max-samples "${SMOKE_SOURCE_ROWS}" \
    --preprocessing-num-workers 1 \
    --preprocessing-batch-size 1000 \
    --overwrite

echo "Smoke output: ${OUTPUT_DIR}"
echo "Smoke log:    ${LOG_FILE}"
cd "${REPO_DIR}"
TOKENIZED_DATA="${TOKENIZED_DATA}" \
MODEL_CHECKPOINT="${MODEL_CHECKPOINT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
MODEL_ID="${MODEL_ID}" \
SEQUENCE_LENGTH="${SMOKE_SEQUENCE_LENGTH}" \
SMOKE=1 \
    bash "${SCRIPT_DIR}/run_sft_trn2_48xlarge.sh" 2>&1 | tee "${LOG_FILE}"

test -f "${OUTPUT_DIR}/trainer_state.json"
echo "SMOKE_TEST_OK"
