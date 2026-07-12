#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Paper-faithful Lightning-OPD SFT curation with native PyTorch on Trainium.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -f "${SCRIPT_DIR}/runtime.env" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/runtime.env"
fi
cd "${REPO_ROOT}"

SFT_SAMPLES_USER="${SFT_SAMPLES-}"
MAX_TOKENS_USER="${MAX_TOKENS-}"
SCALE="${SCALE:-4b}"
SMOKE="${SMOKE:-0}"
FAST_SMOKE="${FAST_SMOKE:-0}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-8B}"
HF_DATASET="${HF_DATASET:-open-thoughts/OpenThoughts3-1.2M}"
SEED="${SEED:-42}"
NUM_CORES="${NUM_CORES:-1}"
TP_SIZE="${TP_SIZE:-1}"
MODE="${MODE:-compile}"
PREFILL_BUCKET="${PREFILL_BUCKET:-512}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VALIDATE="${VALIDATE:-1}"
NATIVE_VENV="${NATIVE_VENV:-${HOME}/workspace/native_venv}"

[[ "${SCALE}" == "4b" ]] || { echo "ERROR: this entrypoint implements the paper's 4B path." >&2; exit 1; }
[[ "${SMOKE}" =~ ^[01]$ && "${FAST_SMOKE}" =~ ^[01]$ ]] || {
    echo "ERROR: SMOKE and FAST_SMOKE must be 0 or 1." >&2; exit 1;
}
[[ "${TP_SIZE}" == "1" ]] || {
    echo "ERROR: pure native PyTorch has no TP here; set TP_SIZE=1." >&2; exit 1;
}
[[ -f "${NATIVE_VENV}/bin/activate" ]] || {
    echo "ERROR: native Beta-3 venv not found: ${NATIVE_VENV}" >&2
    echo "Run trainium/sft_data_generation_native/setup_env.sh first." >&2
    exit 1
}

if [[ "${SMOKE}" == "1" ]]; then
    SFT_SAMPLES="${SFT_SAMPLES:-64}"
    STATE_SUFFIX="-smoke"
else
    SFT_SAMPLES="${SFT_SAMPLES:-300000}"
    STATE_SUFFIX=""
fi
if [[ "${FAST_SMOKE}" == "1" ]]; then
    SFT_SAMPLES="${SFT_SAMPLES_USER:-4}"
    MAX_TOKENS="${MAX_TOKENS_USER:-128}"
    STATE_SUFFIX="-fast-smoke"
fi

# shellcheck disable=SC1091
source "${NATIVE_VENV}/bin/activate"
python -m trainium.sft_data_generation_native.check_env

PROMPT_FILE="data/prompts/openthoughts3_${SFT_SAMPLES}.jsonl"
PARTS_DIR="data/sft_data/qwen3-8b_${SFT_SAMPLES}_arrow"
FINAL_PARQUET="data/sft_data/openthoughts3_${SFT_SAMPLES}_qwen3-8b.parquet"
CURATION_CKPT="agent_artifacts/data/curation_checkpoints/qwen3-8b_${SFT_SAMPLES}"
TRACE_DIR="agent_artifacts/traces"
PIPELINE_STATE="data/.pipeline_state/4b${STATE_SUFFIX}"
mkdir -p "$(dirname "${PROMPT_FILE}")" "${PARTS_DIR}" "${CURATION_CKPT}" "${TRACE_DIR}" "${PIPELINE_STATE}"

echo "=== Native TorchNeuron SFT curation ==="
echo "teacher=${TEACHER_MODEL} dataset=${HF_DATASET} samples=${SFT_SAMPLES} seed=${SEED}"
echo "mode=${MODE} cores=${NUM_CORES} batch=${BATCH_SIZE} prefill=${PREFILL_BUCKET} max_tokens=${MAX_TOKENS}"
echo "sampling: temperature=${TEMPERATURE} top_p=${TOP_P}"

if [[ ! -f "${PROMPT_FILE}" ]]; then
    python scripts/prepare_sft_prompts.py \
        --hf-dataset "${HF_DATASET}" \
        --output "${PROMPT_FILE}" \
        --num-samples "${SFT_SAMPLES}" \
        --seed "${SEED}"
fi

if [[ "${VALIDATE}" == "1" ]]; then
    TORCH_LOGS="graph_breaks,recompiles" \
    python -m trainium.sft_data_generation_native.validate_native \
        --model "${TEACHER_MODEL}" \
        --output "${TRACE_DIR}/native_validation.json"
fi

TORCH_LOGS="graph_breaks,recompiles" \
TEACHER_MODEL="${TEACHER_MODEL}" \
SFT_PROMPTS="${PROMPT_FILE}" \
OUTPUT_DIR="${PARTS_DIR}" \
NUM_CORES="${NUM_CORES}" \
TP_SIZE=1 \
bash trainium/step1_generate_sft_data.sh \
    --device neuron \
    --mode "${MODE}" \
    --prefill-bucket "${PREFILL_BUCKET}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --checkpoint-dir "${CURATION_CKPT}"

python data_curation/merge.py \
    --input-dir "${PARTS_DIR}" \
    --output "${FINAL_PARQUET}" \
    --max-tokens "${MAX_TOKENS}"

python - "${FINAL_PARQUET}" "${SFT_SAMPLES}" <<'PY'
import sys
import pyarrow.parquet as pq

path, expected = sys.argv[1], int(sys.argv[2])
parquet = pq.ParquetFile(path)
columns = set(parquet.schema_arrow.names)
if columns != {"messages", "tokens"}:
    raise SystemExit(f"unexpected schema: {sorted(columns)}")
if parquet.metadata.num_rows != expected:
    raise SystemExit(f"expected {expected} rows, got {parquet.metadata.num_rows}")
print(f"validated {path}: {expected} rows, messages+tokens schema")
PY

touch "${PIPELINE_STATE}/prompts.done" "${PIPELINE_STATE}/sft_data.done" "${PIPELINE_STATE}/sft_merge.done"
cat > "${TRACE_DIR}/port_summary.md" <<EOF
# Native PyTorch SFT data-curation run

- Model: ${TEACHER_MODEL}
- Dataset: ${HF_DATASET}, ${SFT_SAMPLES} prompts, seed ${SEED}
- Device: neuron; dtype: bfloat16; mode: ${MODE}
- Sampling: temperature ${TEMPERATURE}, top-p ${TOP_P}, max new tokens ${MAX_TOKENS}
- Static shapes: batch ${BATCH_SIZE}, prefill bucket ${PREFILL_BUCKET}, StaticCache
- CPU fallback: not enabled
- Output: ${FINAL_PARQUET}
- Validation: $(if [[ "${VALIDATE}" == "1" ]]; then echo "CPU fp32 vs eager+compiled Neuron greedy token gate passed"; else echo "skipped by VALIDATE=0"; fi)
EOF

echo "SFT curation complete: ${FINAL_PARQUET}"
