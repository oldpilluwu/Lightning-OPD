#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CURATION_ENV="${CURATION_ENV:-qwen35-curation}"
SFT_ENV="${SFT_ENV:-qwen35-sft}"
TRAIN_ENV="${TRAIN_ENV:-qwen35-train}"

CONDA_BIN="${CONDA_BIN:-conda}"

have_env() {
    "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "$1"
}

create_env() {
    local env_name="$1"
    if have_env "${env_name}"; then
        echo "Conda env exists: ${env_name}"
    else
        "${CONDA_BIN}" create -n "${env_name}" "python=${PYTHON_VERSION}" -y
    fi
}

echo "Repository: ${ROOT_DIR}"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-all}"

create_env "${CURATION_ENV}"
create_env "${SFT_ENV}"
create_env "${TRAIN_ENV}"

echo "Installing curation env..."
"${CONDA_BIN}" run -n "${CURATION_ENV}" python -m pip install --upgrade pip
"${CONDA_BIN}" run -n "${CURATION_ENV}" python -m pip install \
    "vllm" "transformers>=4.57.0" "pyarrow" "pandas" "tqdm" "datasets" "accelerate"

echo "Installing SFT env..."
"${CONDA_BIN}" run -n "${SFT_ENV}" python -m pip install --upgrade pip
"${CONDA_BIN}" run -n "${SFT_ENV}" python -m pip install \
    "llamafactory" "torch" "transformers>=4.57.0" "datasets" "accelerate" "deepspeed" \
    "liger-kernel" "wandb" "pandas" "pyarrow" "tqdm"

echo "Installing training/sglang env..."
"${CONDA_BIN}" run -n "${TRAIN_ENV}" python -m pip install --upgrade pip
"${CONDA_BIN}" run -n "${TRAIN_ENV}" python -m pip install -r requirements.txt
"${CONDA_BIN}" run -n "${TRAIN_ENV}" python -m pip install \
    "sglang[all]" "vllm" "transformers>=4.57.0" "tensorboard" "aiohttp" "pandas" "pyarrow" "tqdm"
"${CONDA_BIN}" run -n "${TRAIN_ENV}" python -m pip install -e .

echo "Setup complete."
echo "Envs:"
echo "  curation: ${CURATION_ENV}"
echo "  sft:      ${SFT_ENV}"
echo "  train:    ${TRAIN_ENV}"
