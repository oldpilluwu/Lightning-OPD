#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CURATION_ENV="${CURATION_ENV:-qwen35-curation}"
SFT_ENV="${SFT_ENV:-qwen35-sft}"
TRAIN_ENV="${TRAIN_ENV:-qwen35-train}"

CONDA_BIN="${CONDA_BIN:-conda}"
MINICONDA_DIR="${MINICONDA_DIR:-${HOME}/miniconda3}"
SETUP_LOG_DIR="${SETUP_LOG_DIR:-logs/qwen35_2b_9b/setup}"

ensure_conda() {
    if command -v "${CONDA_BIN}" >/dev/null 2>&1; then
        return
    fi

    if [[ -x "${MINICONDA_DIR}/bin/conda" ]]; then
        CONDA_BIN="${MINICONDA_DIR}/bin/conda"
        return
    fi

    echo "conda not found. Installing Miniconda to ${MINICONDA_DIR}..."
    local installer="/tmp/miniconda.sh"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "${installer}"
    elif command -v wget >/dev/null 2>&1; then
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "${installer}"
    else
        echo "Neither curl nor wget is available. Install one of them or install Miniconda manually." >&2
        exit 1
    fi

    bash "${installer}" -b -p "${MINICONDA_DIR}"
    CONDA_BIN="${MINICONDA_DIR}/bin/conda"
    "${CONDA_BIN}" config --set auto_activate_base false
}

have_env() {
    "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "$1"
}

create_env() {
    local env_name="$1"
    if have_env "${env_name}"; then
        echo "Conda env exists: ${env_name}"
    else
        "${CONDA_BIN}" create \
            --override-channels \
            -c conda-forge \
            -n "${env_name}" \
            "python=${PYTHON_VERSION}" \
            -y
    fi
}

run_in_env() {
    local env_name="$1"
    local log_name="$2"
    shift 2
    mkdir -p "${SETUP_LOG_DIR}"
    echo "[$(date '+%F %T')] ${env_name}: $*"
    "${CONDA_BIN}" run -n "${env_name}" bash -lc "$*" 2>&1 | tee -a "${SETUP_LOG_DIR}/${log_name}.log"
}

pip_install() {
    local env_name="$1"
    local log_name="$2"
    shift 2
    run_in_env "${env_name}" "${log_name}" \
        "python -m pip install --progress-bar on --retries 5 --timeout 120 $*"
}

echo "Repository: ${ROOT_DIR}"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-all}"

ensure_conda
echo "Using conda: ${CONDA_BIN}"
echo "Creating envs from conda-forge with --override-channels to avoid Anaconda default-channel ToS prompts."
echo "Setup logs: ${SETUP_LOG_DIR}"

create_env "${CURATION_ENV}"
create_env "${SFT_ENV}"
create_env "${TRAIN_ENV}"

echo "Installing curation env..."
run_in_env "${CURATION_ENV}" "curation_00_pip" "python -m pip install --upgrade pip"
pip_install "${CURATION_ENV}" "curation_01_core" \
    "transformers>=4.57.0" "pyarrow" "pandas" "tqdm" "datasets" "accelerate" "huggingface_hub[cli]"
pip_install "${CURATION_ENV}" "curation_02_vllm" "vllm"

echo "Installing SFT env..."
run_in_env "${SFT_ENV}" "sft_00_pip" "python -m pip install --upgrade pip"
pip_install "${SFT_ENV}" "sft_01_core" \
    "torch" "transformers>=4.57.0" "datasets" "accelerate" "pandas" "pyarrow" "tqdm" "wandb" "huggingface_hub[cli]"
pip_install "${SFT_ENV}" "sft_02_training" "llamafactory" "deepspeed" "liger-kernel"

echo "Installing training/sglang env..."
run_in_env "${TRAIN_ENV}" "train_00_pip" "python -m pip install --upgrade pip"
run_in_env "${TRAIN_ENV}" "train_01_requirements" \
    "python -m pip install --progress-bar on --retries 5 --timeout 120 -r requirements.txt"
pip_install "${TRAIN_ENV}" "train_02_core" \
    "transformers>=4.57.0" "tensorboard" "aiohttp" "pandas" "pyarrow" "tqdm" "huggingface_hub[cli]"
pip_install "${TRAIN_ENV}" "train_03_vllm" "vllm"
pip_install "${TRAIN_ENV}" "train_04_sglang" "sglang[all]"
run_in_env "${TRAIN_ENV}" "train_05_editable" "python -m pip install -e ."

echo "Setup complete."
echo "Envs:"
echo "  curation: ${CURATION_ENV}"
echo "  sft:      ${SFT_ENV}"
echo "  train:    ${TRAIN_ENV}"
