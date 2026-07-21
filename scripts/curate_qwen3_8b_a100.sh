#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# End-to-end data curation for a single A100 80GB server.
#
# Run from the repository root:
#   bash scripts/curate_qwen3_8b_a100.sh
#
# The script does everything needed for SFT data curation:
#   1. create/reuse a local Python virtualenv
#   2. install data-curation dependencies
#   3. download Qwen3-8B and the prompt dataset from Hugging Face
#   4. extract prompt-only JSONL
#   5. generate teacher responses with vLLM
#   6. merge rank-local Arrow shards into one parquet dataset
#
# Useful optional overrides:
#   NUM_SAMPLES=10000 bash scripts/curate_qwen3_8b_a100.sh
#   FORCE=1 BATCH_SIZE=128 bash scripts/curate_qwen3_8b_a100.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

log() {
    printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
    printf '\nERROR: %s\n' "$*" >&2
    exit 1
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

hf_download() {
    local repo="$1"
    local local_dir="$2"

    if command_exists hf; then
        hf download "${repo}" --local-dir "${local_dir}"
    elif command_exists huggingface-cli; then
        huggingface-cli download "${repo}" --local-dir "${local_dir}"
    else
        die "Neither 'hf' nor 'huggingface-cli' was found after installing huggingface_hub."
    fi
}

detect_python() {
    if command_exists python3.10; then
        echo "python3.10"
    elif command_exists python3; then
        echo "python3"
    elif command_exists python; then
        echo "python"
    else
        die "Python 3.10+ was not found on PATH."
    fi
}

python_version_short() {
    "${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
}

install_python_venv_package() {
    if ! command_exists apt-get; then
        return 1
    fi

    local py_ver
    py_ver="$(python_version_short)"
    local package="python${py_ver}-venv"

    log "Installing ${package} so Python can create virtualenvs"
    if [[ "$(id -u)" -eq 0 ]]; then
        DEBIAN_FRONTEND=noninteractive apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y "${package}"
    elif command_exists sudo; then
        sudo DEBIAN_FRONTEND=noninteractive apt-get update
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${package}"
    else
        return 1
    fi
}

create_virtualenv() {
    local venv_dir="$1"

    if [[ -f "${venv_dir}/bin/activate" ]]; then
        return 0
    fi

    rm -rf "${venv_dir}"
    if "${PYTHON_BIN}" -m venv "${venv_dir}"; then
        return 0
    fi

    log "Standard venv creation failed; attempting to install the missing system venv package"
    rm -rf "${venv_dir}"
    if install_python_venv_package && "${PYTHON_BIN}" -m venv "${venv_dir}"; then
        return 0
    fi

    log "System venv package path failed; trying virtualenv via pip"
    rm -rf "${venv_dir}"
    if "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
        "${PYTHON_BIN}" -m pip install --user --upgrade virtualenv
        "${PYTHON_BIN}" -m virtualenv "${venv_dir}"
        return 0
    fi

    die "Could not create a virtualenv. Install python$(python_version_short)-venv, then rerun this script."
}

detect_gpu_count() {
    if command_exists nvidia-smi; then
        local count
        count="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "${count}" =~ ^[0-9]+$ ]] && [[ "${count}" -gt 0 ]]; then
            echo "${count}"
            return
        fi
    fi
    echo "1"
}

require_gpu() {
    if ! command_exists nvidia-smi; then
        die "nvidia-smi was not found. Run this on the A100 server with NVIDIA drivers installed."
    fi
    nvidia-smi >/dev/null || die "nvidia-smi failed. Check the NVIDIA driver/CUDA installation."
}

# Inputs and outputs.
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-8B}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/qwen3-8b}"
HF_DATASET="${HF_DATASET:-open-thoughts/OpenThoughts3-1.2M}"
NUM_SAMPLES="${NUM_SAMPLES:-300000}"
SEED="${SEED:-42}"

PROMPTS_FILE="${PROMPTS_FILE:-${ROOT_DIR}/data/prompts/openthoughts3_${NUM_SAMPLES}.jsonl}"
RAW_OUTPUT_DIR="${RAW_OUTPUT_DIR:-${ROOT_DIR}/data/sft_data/qwen3_8b_a100_raw}"
FINAL_PARQUET="${FINAL_PARQUET:-${ROOT_DIR}/data/sft_data/openthoughts3_${NUM_SAMPLES}_qwen3-8b.parquet}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/data/.pipeline_state/qwen3_8b_a100}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-curation}"

# vLLM defaults tuned for Qwen3-8B on an A100 80GB. Lower BATCH_SIZE or
# GPU_MEMORY_UTILIZATION if the local driver/runtime leaves less free memory.
NUM_GPUS="${NUM_GPUS:-$(detect_gpu_count)}"
TP_SIZE="${TP_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
SWAP_SPACE="${SWAP_SPACE:-16}"
DTYPE="${DTYPE:-bfloat16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
NUM_RESPONSES="${NUM_RESPONSES:-1}"

FORCE="${FORCE:-0}"
FORCE_REINSTALL="${FORCE_REINSTALL:-0}"
CLEAN_RAW="${CLEAN_RAW:-0}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if (( NUM_GPUS < TP_SIZE )); then
    die "NUM_GPUS (${NUM_GPUS}) must be >= TP_SIZE (${TP_SIZE})."
fi
if (( NUM_GPUS % TP_SIZE != 0 )); then
    die "NUM_GPUS (${NUM_GPUS}) must be divisible by TP_SIZE (${TP_SIZE})."
fi

require_gpu

log "GPU inventory"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

PYTHON_BIN="$(detect_python)"
log "Setting up virtualenv at ${VENV_DIR}"
create_virtualenv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required.")
print(f"Using Python {sys.version.split()[0]}")
PY

DEPS_MARKER="${VENV_DIR}/.curation-deps-installed"
if [[ "${FORCE_REINSTALL}" == "1" || ! -f "${DEPS_MARKER}" ]]; then
    log "Installing curation dependencies"
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install --upgrade \
        "vllm" \
        "transformers" \
        "datasets" \
        "huggingface_hub[hf_transfer]" \
        "pyarrow" \
        "pandas" \
        "tqdm"
    date '+%Y-%m-%d %H:%M:%S' > "${DEPS_MARKER}"
else
    log "Dependencies already installed; set FORCE_REINSTALL=1 to reinstall"
fi

mkdir -p "$(dirname "${PROMPTS_FILE}")" "${RAW_OUTPUT_DIR}" "$(dirname "${FINAL_PARQUET}")" "${CHECKPOINT_DIR}" "${MODEL_DIR}"

if [[ "${FORCE}" == "1" ]]; then
    log "FORCE=1: removing previous generated artifacts"
    rm -rf "${RAW_OUTPUT_DIR}" "${CHECKPOINT_DIR}" "${FINAL_PARQUET}"
    mkdir -p "${RAW_OUTPUT_DIR}" "${CHECKPOINT_DIR}" "$(dirname "${FINAL_PARQUET}")"
fi

if [[ -f "${FINAL_PARQUET}" && "${FORCE}" != "1" ]]; then
    log "Final parquet already exists: ${FINAL_PARQUET}"
    log "Nothing to do. Set FORCE=1 to regenerate from scratch."
    exit 0
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    log "Downloading model ${MODEL_REPO} to ${MODEL_DIR}"
    hf_download "${MODEL_REPO}" "${MODEL_DIR}"
else
    log "Model already present at ${MODEL_DIR}"
fi

if [[ ! -f "${PROMPTS_FILE}" || "${FORCE}" == "1" ]]; then
    log "Downloading and processing prompt dataset ${HF_DATASET}"
    python scripts/prepare_sft_prompts.py \
        --hf-dataset "${HF_DATASET}" \
        --output "${PROMPTS_FILE}" \
        --num-samples "${NUM_SAMPLES}" \
        --seed "${SEED}"
else
    log "Prompt file already present: ${PROMPTS_FILE}"
fi

log "Starting vLLM generation with Qwen3-8B"
log "NUM_GPUS=${NUM_GPUS} TP_SIZE=${TP_SIZE} BATCH_SIZE=${BATCH_SIZE} MAX_TOKENS=${MAX_TOKENS}"
TEACHER_MODEL="${MODEL_DIR}" \
SFT_PROMPTS="${PROMPTS_FILE}" \
OUTPUT_DIR="${RAW_OUTPUT_DIR}" \
NUM_GPUS="${NUM_GPUS}" \
TP_SIZE="${TP_SIZE}" \
bash scripts/generate_sft_data.sh \
    --batch-size "${BATCH_SIZE}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --num-responses "${NUM_RESPONSES}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --dtype "${DTYPE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --swap-space "${SWAP_SPACE}" \
    --enable-prefix-caching

log "Merging generated Arrow shards into ${FINAL_PARQUET}"
python data_curation/merge.py \
    --input-dir "${RAW_OUTPUT_DIR}" \
    --output "${FINAL_PARQUET}"

if [[ "${CLEAN_RAW}" == "1" ]]; then
    log "CLEAN_RAW=1: removing raw Arrow shards"
    find "${RAW_OUTPUT_DIR}" -name "*.arrow" -delete
    find "${RAW_OUTPUT_DIR}" -type d -name "rank*" -prune -exec rm -rf {} +
fi

log "Done"
log "Final dataset: ${FINAL_PARQUET}"
