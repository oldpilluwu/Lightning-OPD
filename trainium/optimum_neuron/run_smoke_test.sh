#!/usr/bin/env bash
# Prepare 1,000 examples and run two full-model Qwen3 optimizer steps on TP8.
# Trn2 supports 16 workers as the smallest worker count divisible by TP8.

set -euo pipefail

unset PYTHONPATH
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PREP_DIR="${PREP_DIR:-$(cd -- "${REPO_DIR}/.." && pwd)/lightning-opd-prep}"
SFT_DATA="${SFT_DATA:-${PREP_DIR}/sft_data/openthoughts3_300000_qwen3-8b.parquet}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Base}"
EXPECTED_ROWS="${EXPECTED_ROWS:-300000}"
SMOKE_SEQUENCE_LENGTH="${SMOKE_SEQUENCE_LENGTH:-1024}"
SMOKE_SOURCE_ROWS="${SMOKE_SOURCE_ROWS:-1000}"
SMOKE_ROOT="${SMOKE_ROOT:-${HOME}/lightning-opd-smoke}"
MODEL_CACHE_KEY="${MODEL_ID//\//_}"
MODEL_CACHE_KEY="${MODEL_CACHE_KEY//:/_}"
SOURCE_CACHE_KEY="$(basename -- "${SFT_DATA}" .parquet)"
TOKENIZED_DATA="${TOKENIZED_DATA:-${SMOKE_ROOT}/${SOURCE_CACHE_KEY}-${MODEL_CACHE_KEY}-rows${SMOKE_SOURCE_ROWS}-seq${SMOKE_SEQUENCE_LENGTH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SMOKE_ROOT}/run-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_FILE="${OUTPUT_DIR}.log"

if [[ ! -e "${SFT_DATA}" ]]; then
    echo "ERROR: generated SFT dataset not found: ${SFT_DATA}" >&2
    echo "Expected the data-generation checkout next to Lightning-OPD:" >&2
    echo "  ${PREP_DIR}/sft_data/openthoughts3_300000_qwen3-8b.parquet" >&2
    echo "Set SFT_DATA explicitly only if the Parquet is stored elsewhere." >&2
    exit 1
fi

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
    SFT_DATA="${SFT_DATA}" \
    MODEL_ID="${MODEL_ID}" \
    EXPECTED_ROWS="${EXPECTED_ROWS}" \
    SMOKE_SOURCE_ROWS="${SMOKE_SOURCE_ROWS}" \
    SMOKE_SEQUENCE_LENGTH="${SMOKE_SEQUENCE_LENGTH}" \
    TOKENIZED_DATA="${TOKENIZED_DATA}" \
    python - <<'PY'
import json
import os
from pathlib import Path

metadata_path = Path(os.environ["TOKENIZED_DATA"]) / "reproduction_metadata.json"
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
expected = {
    "model_id": os.environ["MODEL_ID"],
    "source_rows": int(os.environ["EXPECTED_ROWS"]),
    "selected_source_rows": int(os.environ["SMOKE_SOURCE_ROWS"]),
    "cutoff_len": int(os.environ["SMOKE_SEQUENCE_LENGTH"]),
    "preprocessing_num_workers": 1,
    "preprocessing_batch_size": 1000,
}
mismatches = {
    key: (metadata.get(key), value)
    for key, value in expected.items()
    if metadata.get(key) != value
}
source = str(Path(os.environ["SFT_DATA"]).expanduser().resolve())
if metadata.get("source_files") != [source]:
    mismatches["source_files"] = (metadata.get("source_files"), [source])
if mismatches:
    raise SystemExit(
        f"Refusing to reuse stale smoke preprocessing at {metadata_path}: {mismatches}"
    )
PY
    echo "Reusing verified smoke dataset: ${TOKENIZED_DATA}"
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
