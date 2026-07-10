#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Generate the paper-faithful Qwen3-8B SFT dataset on AWS Trainium, then mark
# run_pipeline.sh's prompt/SFT-generation/SFT-merge stages complete so the main
# pipeline resumes at SFT training.
#
# Default (paper data scale):
#   bash trainium/sft_data_generation/generate_sft_data.sh
#
# Faithful smoke run (same generation settings, 64 prompts):
#   SMOKE=1 bash trainium/sft_data_generation/generate_sft_data.sh
#
# Resume the main pipeline afterwards with the same overrides:
#   SCALE=4b SFT_SAMPLES=300000 bash trainium/run_pipeline.sh
#   SCALE=4b SMOKE=1 SFT_SAMPLES=64 bash trainium/run_pipeline.sh
#
# The PyTorch 2.9 NxD-Inference venv is supported when the vllm-neuron plugin
# is installed. Override INFER_VENV to use the preconfigured vLLM venv instead:
#   INFER_VENV=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16 \
#     bash trainium/sft_data_generation/generate_sft_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${SCRIPT_DIR}/runtime.env" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/runtime.env"
fi

cd "${REPO_ROOT}"

SCALE="${SCALE:-4b}"
SMOKE="${SMOKE:-0}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-8B}"
HF_DATASET="${HF_DATASET:-open-thoughts/OpenThoughts3-1.2M}"
SEED="${SEED:-42}"
NUM_CORES="${NUM_CORES:-4}"
TP_SIZE="${TP_SIZE:-${NUM_CORES}}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
NUM_RESPONSES="${NUM_RESPONSES:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-18432}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
# This is the outer request batch, not the compiled device batch. Keeping more
# prompts queued than MAX_NUM_SEQS lets continuous batching refill freed slots.
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-${MAX_NUM_SEQS}}"
DEVICE_HBM_GIB="${DEVICE_HBM_GIB:-96}"
RUNTIME_RESERVE_GIB="${RUNTIME_RESERVE_GIB:-32}"
INFER_VENV="${INFER_VENV:-/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference}"

if [[ "${SCALE}" != "4b" ]]; then
    echo "ERROR: this entrypoint implements the paper's 4B student data path only:" >&2
    echo "       OpenThoughts3 prompts + Qwen/Qwen3-8B teacher. Set SCALE=4b." >&2
    exit 1
fi

if [[ "${SMOKE}" != "0" && "${SMOKE}" != "1" ]]; then
    echo "ERROR: SMOKE must be 0 or 1 (got '${SMOKE}')." >&2
    exit 1
fi

if [[ "${SMOKE}" == "1" ]]; then
    SFT_SAMPLES="${SFT_SAMPLES:-64}"
    STATE_SUFFIX="-smoke"
else
    SFT_SAMPLES="${SFT_SAMPLES:-300000}"
    STATE_SUFFIX=""
fi

if (( NUM_CORES <= 0 || TP_SIZE <= 0 || NUM_CORES % TP_SIZE != 0 )); then
    echo "ERROR: NUM_CORES (${NUM_CORES}) and TP_SIZE (${TP_SIZE}) must be positive," >&2
    echo "       and NUM_CORES must be divisible by TP_SIZE." >&2
    exit 1
fi

if (( BATCH_SIZE <= 0 || MAX_NUM_SEQS <= 0 )); then
    echo "ERROR: BATCH_SIZE and MAX_NUM_SEQS must be positive." >&2
    exit 1
fi

if (( NUM_GPU_BLOCKS_OVERRIDE != MAX_NUM_SEQS )); then
    echo "ERROR: with prefix caching disabled, vllm-neuron requires" >&2
    echo "       NUM_GPU_BLOCKS_OVERRIDE (${NUM_GPU_BLOCKS_OVERRIDE}) ==" >&2
    echo "       MAX_NUM_SEQS (${MAX_NUM_SEQS})." >&2
    exit 1
fi

if [[ ! -f "${INFER_VENV}/bin/activate" ]]; then
    echo "ERROR: inference venv not found: ${INFER_VENV}" >&2
    echo "Set INFER_VENV to one of the installed Neuron inference environments." >&2
    exit 1
fi

if ! command -v neuron-ls >/dev/null 2>&1; then
    echo "ERROR: neuron-ls is unavailable; run this on a Neuron DLAMI/instance." >&2
    exit 1
fi

echo "=== Trainium SFT data generation ==="
echo "  Teacher:        ${TEACHER_MODEL}"
echo "  Source dataset: ${HF_DATASET}"
echo "  Prompt count:   ${SFT_SAMPLES}"
echo "  NeuronCores:    ${NUM_CORES} (TP=${TP_SIZE})"
echo "  Inference venv: ${INFER_VENV}"
echo "  Sampling:       temperature=${TEMPERATURE}, top_p=${TOP_P}, max_tokens=${MAX_TOKENS}"
echo "  Concurrency:    batch_size=${BATCH_SIZE}, max_num_seqs=${MAX_NUM_SEQS}"
neuron-ls

# shellcheck disable=SC1091
source "${INFER_VENV}/bin/activate"

# The upstream plugin uses this selector for its NxD-Inference platform.
export VLLM_NEURON_FRAMEWORK="${VLLM_NEURON_FRAMEWORK:-neuronx-distributed-inference}"

# Qwen3-8B has 36 layers, 8 KV heads, and head_dim=128. With BF16 K/V,
# one fully allocated sequence uses:
#   2 (K+V) * 36 * 8 * 128 * 2 bytes * MAX_MODEL_LEN.
# NxD Inference allocates a contiguous cache for every compiled sequence slot.
# Include one model replica per TP worker group and reserve HBM for graphs,
# activations, runtime buffers, and temporary compilation allocations.
python - "${NUM_CORES}" "${TP_SIZE}" "${MAX_MODEL_LEN}" "${MAX_NUM_SEQS}" \
    "${DEVICE_HBM_GIB}" "${RUNTIME_RESERVE_GIB}" <<'PY'
import sys

num_cores, tp_size, max_len, max_seqs = map(int, sys.argv[1:5])
hbm_gib, reserve_gib = map(float, sys.argv[5:7])
workers = num_cores // tp_size
bytes_per_token = 2 * 36 * 8 * 128 * 2
kv_gib_per_worker = bytes_per_token * max_len * max_seqs / 2**30
# Qwen3-8B BF16 weights are approximately 16 GiB per model replica before TP.
weights_gib_per_worker = 16.0
estimated_gib = workers * (weights_gib_per_worker + kv_gib_per_worker)
usable_gib = hbm_gib - reserve_gib

print(
    "Estimated static allocation: "
    f"{estimated_gib:.1f} GiB across {workers} worker(s) "
    f"(weights {workers * weights_gib_per_worker:.1f} GiB + "
    f"KV cache {workers * kv_gib_per_worker:.1f} GiB); "
    f"budget {usable_gib:.1f} GiB after {reserve_gib:.1f} GiB reserve."
)
if estimated_gib > usable_gib:
    raise SystemExit(
        "Estimated model + KV allocation exceeds the conservative HBM budget. "
        "Lower MAX_NUM_SEQS/BATCH_SIZE, increase TP_SIZE, or set DEVICE_HBM_GIB "
        "and RUNTIME_RESERVE_GIB for the actual instance after measurement."
    )
PY

if ! python -c "import vllm" >/dev/null 2>&1; then
    cat >&2 <<EOF
ERROR: vLLM is not installed in ${INFER_VENV}.

Set up the PyTorch 2.9 NxD-Inference environment with:

  bash trainium/sft_data_generation/setup_env.sh

Alternatively, use the already configured environment:

  INFER_VENV=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16 \\
    bash trainium/sft_data_generation/generate_sft_data.sh
EOF
    exit 1
fi

if [[ "$(basename "${INFER_VENV}")" == *nxd_inference* ]] \
    && ! python -c "import vllm_neuron" >/dev/null 2>&1; then
    echo "ERROR: ${INFER_VENV} has vLLM but not the vllm-neuron platform plugin." >&2
    echo "Install https://github.com/vllm-project/vllm-neuron with the AWS Neuron" >&2
    echo "package index, or set INFER_VENV to" >&2
    echo "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16." >&2
    exit 1
fi

python - <<'PY'
import importlib.metadata as metadata
import vllm

print(f"vLLM import OK: {vllm.__version__}")
for package in ("vllm-neuron", "neuronx-distributed-inference", "torch-neuronx"):
    try:
        print(f"{package}: {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package}: distribution metadata not found")
PY

PROMPT_DIR="data/prompts"
PROMPT_FILE="${PROMPT_DIR}/openthoughts3_${SFT_SAMPLES}.jsonl"
SFT_ROOT="data/sft_data"
PARTS_DIR="${SFT_ROOT}/qwen3-8b_${SFT_SAMPLES}_arrow"
FINAL_PARQUET="${SFT_ROOT}/openthoughts3_${SFT_SAMPLES}_qwen3-8b.parquet"
CURATION_CKPT="data/.curation_checkpoints/qwen3-8b_${SFT_SAMPLES}"
LOCAL_STATE="data/.sft_data_generation/qwen3-8b_${SFT_SAMPLES}"
PIPELINE_STATE="data/.pipeline_state/4b${STATE_SUFFIX}"

mkdir -p "${PROMPT_DIR}" "${SFT_ROOT}" "${PARTS_DIR}" \
    "${CURATION_CKPT}" "${LOCAL_STATE}" "${PIPELINE_STATE}"

validate_prompts() {
    python - "${PROMPT_FILE}" "${SFT_SAMPLES}" <<'PY'
import json
import sys

path = sys.argv[1]
expected = int(sys.argv[2])
count = 0
with open(path, encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        row = json.loads(line)
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise SystemExit(f"{path}:{line_number}: missing non-empty prompt list")
        count += 1
if count != expected:
    raise SystemExit(f"{path}: found {count} prompts, expected {expected}")
print(f"Prompt validation passed: {count} rows")
PY
}

validate_parquet() {
    python - "${FINAL_PARQUET}" <<'PY'
import sys
import pyarrow.parquet as pq

path = sys.argv[1]
parquet = pq.ParquetFile(path)
columns = set(parquet.schema_arrow.names)
required = {"messages", "tokens"}
missing = required - columns
if missing:
    raise SystemExit(f"{path}: missing columns: {sorted(missing)}")
if parquet.metadata.num_rows <= 0:
    raise SystemExit(f"{path}: contains no rows")
print(f"Parquet validation passed: {parquet.metadata.num_rows} rows, columns={sorted(columns)}")
PY
}

if [[ -f "${FINAL_PARQUET}" ]] && validate_parquet; then
    echo ">>> Final SFT parquet already exists; generation is complete."
else
    if [[ -f "${PROMPT_FILE}" ]] && validate_prompts; then
        echo ">>> Prompt dataset already exists; skipping prompt preparation."
    else
        echo ">>> Preparing ${SFT_SAMPLES} OpenThoughts3 prompts (seed=${SEED})..."
        rm -f "${PROMPT_FILE}.tmp"
        python scripts/prepare_sft_prompts.py \
            --hf-dataset "${HF_DATASET}" \
            --output "${PROMPT_FILE}.tmp" \
            --num-samples "${SFT_SAMPLES}" \
            --seed "${SEED}"
        mv "${PROMPT_FILE}.tmp" "${PROMPT_FILE}"
        validate_prompts
    fi
    touch "${LOCAL_STATE}/prompts.done"

    if [[ -f "${LOCAL_STATE}/sft_data.done" ]] \
        && find "${PARTS_DIR}" -name "*.arrow" -print -quit | grep -q .; then
        echo ">>> Completed Arrow shards already exist; skipping vLLM generation."
    else
        echo ">>> Generating Qwen3-8B responses with vLLM on Neuron..."
        TEACHER_MODEL="${TEACHER_MODEL}" \
        SFT_PROMPTS="${PROMPT_FILE}" \
        OUTPUT_DIR="${PARTS_DIR}" \
        NUM_CORES="${NUM_CORES}" \
        TP_SIZE="${TP_SIZE}" \
        bash trainium/step1_generate_sft_data.sh \
            --max-tokens "${MAX_TOKENS}" \
            --temperature "${TEMPERATURE}" \
            --top-p "${TOP_P}" \
            --num-responses "${NUM_RESPONSES}" \
            --batch-size "${BATCH_SIZE}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --max-num-seqs "${MAX_NUM_SEQS}" \
            --num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}" \
            --checkpoint-dir "${CURATION_CKPT}"
        touch "${LOCAL_STATE}/sft_data.done"
    fi

    echo ">>> Merging Arrow shards into ${FINAL_PARQUET}..."
    python data_curation/merge.py \
        --input-dir "${PARTS_DIR}" \
        --output "${FINAL_PARQUET}" \
        --max-tokens "${MAX_TOKENS}"
    validate_parquet
    touch "${LOCAL_STATE}/sft_merge.done"
fi

# These are the exact stage names used by trainium/run_pipeline.sh. They are
# written only after the final parquet has passed validation.
touch "${PIPELINE_STATE}/prompts.done"
touch "${PIPELINE_STATE}/sft_data.done"
touch "${PIPELINE_STATE}/sft_merge.done"

cat > "${PIPELINE_STATE}/sft_generation.env" <<EOF
SCALE=4b
SMOKE=${SMOKE}
SFT_SAMPLES=${SFT_SAMPLES}
SFT_PARQUET=${FINAL_PARQUET}
TEACHER_MODEL=${TEACHER_MODEL}
EOF

echo
echo "=== SFT data generation complete ==="
echo "Parquet: ${FINAL_PARQUET}"
echo "run_pipeline.sh will resume at SFT training with:"
echo "  SCALE=4b SMOKE=${SMOKE} SFT_SAMPLES=${SFT_SAMPLES} bash trainium/run_pipeline.sh"
