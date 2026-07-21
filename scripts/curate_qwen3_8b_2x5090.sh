#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# End-to-end Qwen3-8B data curation for one server with 2x RTX 5090 GPUs.
#
# Run from the repository root:
#   bash scripts/curate_qwen3_8b_2x5090.sh
#
# The script does everything needed for SFT data curation:
#   1. create/reuse a local Python 3.12 virtualenv with uv
#   2. install the latest data-curation dependencies
#   3. download Qwen3-8B and the prompt dataset from Hugging Face
#   4. extract prompt-only JSONL
#   5. generate teacher responses with vLLM using both GPUs
#   6. merge rank-local Arrow shards into one parquet dataset

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

apt_install() {
    if ! command_exists apt-get; then
        return 1
    fi

    if [[ "$(id -u)" -eq 0 ]]; then
        DEBIAN_FRONTEND=noninteractive apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"
    elif command_exists sudo; then
        sudo DEBIAN_FRONTEND=noninteractive apt-get update
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"
    else
        return 1
    fi
}

ensure_uv() {
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    if command_exists uv; then
        return 0
    fi

    log "Installing latest uv"
    if ! command_exists curl; then
        apt_install curl ca-certificates || die "curl is required to install uv, and automatic apt install failed."
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    command_exists uv || die "uv installation finished, but 'uv' is not on PATH."
}

venv_python_version() {
    local venv_dir="$1"
    "${venv_dir}/bin/python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
}

create_python_env() {
    local venv_dir="$1"

    ensure_uv

    if [[ -f "${venv_dir}/bin/python" ]]; then
        local current_version
        current_version="$(venv_python_version "${venv_dir}" || true)"
        if [[ "${current_version}" == "${PYTHON_VERSION}" ]]; then
            return 0
        fi
        log "Existing virtualenv uses Python ${current_version}; recreating with Python ${PYTHON_VERSION}"
        rm -rf "${venv_dir}"
    elif [[ -d "${venv_dir}" ]]; then
        log "Removing incomplete virtualenv at ${venv_dir}"
        rm -rf "${venv_dir}"
    fi

    log "Creating Python ${PYTHON_VERSION} virtualenv with uv"
    uv venv "${venv_dir}" --python "${PYTHON_VERSION}" --seed --managed-python
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

detect_gpu_count() {
    if command_exists nvidia-smi; then
        local count
        count="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "${count}" =~ ^[0-9]+$ ]] && [[ "${count}" -gt 0 ]]; then
            echo "${count}"
            return
        fi
    fi
    echo "2"
}

require_gpu() {
    if ! command_exists nvidia-smi; then
        die "nvidia-smi was not found. Run this on the 2x5090 server with NVIDIA drivers installed."
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
RAW_OUTPUT_DIR="${RAW_OUTPUT_DIR:-${ROOT_DIR}/data/sft_data/qwen3_8b_2x5090_raw}"
FINAL_PARQUET="${FINAL_PARQUET:-${ROOT_DIR}/data/sft_data/openthoughts3_${NUM_SAMPLES}_qwen3-8b_2x5090.parquet}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/data/.pipeline_state/qwen3_8b_2x5090}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv-curation}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

# RTX 5090 defaults. Qwen3-8B is run as one replica per GPU for throughput.
# If the local CUDA/PyTorch/vLLM stack OOMs, rerun with:
#   BATCH_SIZE=32 MAX_NUM_SEQS=32 GPU_MEMORY_UTILIZATION=0.86 bash scripts/curate_qwen3_8b_2x5090.sh
NUM_GPUS="${NUM_GPUS:-$(detect_gpu_count)}"
TP_SIZE="${TP_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SWAP_SPACE="${SWAP_SPACE:-8}"
DTYPE="${DTYPE:-bfloat16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
NUM_RESPONSES="${NUM_RESPONSES:-1}"

FORCE="${FORCE:-0}"
CLEAN_RAW="${CLEAN_RAW:-0}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"

if (( NUM_GPUS < TP_SIZE )); then
    die "NUM_GPUS (${NUM_GPUS}) must be >= TP_SIZE (${TP_SIZE})."
fi
if (( NUM_GPUS % TP_SIZE != 0 )); then
    die "NUM_GPUS (${NUM_GPUS}) must be divisible by TP_SIZE (${TP_SIZE})."
fi

require_gpu

log "GPU inventory"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

log "Local layout"
echo "  NUM_GPUS=${NUM_GPUS}"
echo "  TP_SIZE=${TP_SIZE}"
echo "  workers=$((NUM_GPUS / TP_SIZE))"

log "Setting up Python ${PYTHON_VERSION} virtualenv at ${VENV_DIR}"
create_python_env "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 is required, got {sys.version.split()[0]}")
print(f"Using Python {sys.version.split()[0]}")
PY

log "Installing/upgrading latest curation dependencies"
uv pip install --upgrade --torch-backend="${UV_TORCH_BACKEND}" \
    "pip" \
    "setuptools" \
    "wheel" \
    "vllm" \
    "transformers" \
    "datasets" \
    "huggingface_hub[hf_transfer]" \
    "pyarrow" \
    "pandas" \
    "tqdm"

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

log "Starting vLLM generation with Qwen3-8B on 2x5090"
log "BATCH_SIZE=${BATCH_SIZE} MAX_TOKENS=${MAX_TOKENS} MAX_MODEL_LEN=${MAX_MODEL_LEN}"
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
