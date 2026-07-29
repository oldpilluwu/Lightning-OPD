#!/usr/bin/env bash
# Prepare the pinned TorchTitan checkout used by the native torch-neuronx Qwen3 tutorial.

set -euo pipefail

export TORCH_DEVICE_BACKEND_AUTOLOAD=0

TORCH_NEURONX_SRC="${TORCH_NEURONX_SRC:-${HOME}/torch-neuronx}"
TORCHTITAN_DIR="${TORCHTITAN_DIR:-${HOME}/torchtitan}"
TORCHTITAN_COMMIT="0a2107f984639e23a0e5b07fc278785345f03b73"
PATCH_FILE="${TORCH_NEURONX_SRC}/docs/torchtitan/qwen3/TorchTitan.diff"

if [[ ! -f "${PATCH_FILE}" ]]; then
    echo "ERROR: missing the torch-neuronx Qwen3 patch: ${PATCH_FILE}" >&2
    exit 1
fi

if [[ ! -d "${TORCHTITAN_DIR}/.git" ]]; then
    git clone https://github.com/pytorch/torchtitan.git "${TORCHTITAN_DIR}"
    git -C "${TORCHTITAN_DIR}" checkout "${TORCHTITAN_COMMIT}"
fi

ACTUAL_COMMIT="$(git -C "${TORCHTITAN_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${TORCHTITAN_COMMIT}" ]]; then
    echo "ERROR: ${TORCHTITAN_DIR} is at ${ACTUAL_COMMIT}, expected ${TORCHTITAN_COMMIT}." >&2
    echo "Use a separate clean checkout or explicitly check out the pinned commit." >&2
    exit 1
fi

if git -C "${TORCHTITAN_DIR}" apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
    git -C "${TORCHTITAN_DIR}" apply "${PATCH_FILE}"
elif git -C "${TORCHTITAN_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
    echo "TorchTitan Qwen3 Neuron patch is already applied."
else
    echo "ERROR: the Qwen3 Neuron patch cannot be applied cleanly." >&2
    git -C "${TORCHTITAN_DIR}" status --short >&2
    exit 1
fi

python3 - <<'PY'
import torch
import torch_neuronx

assert torch.device("neuron").type == "neuron"
assert "neuron" in getattr(torch.distributed.Backend, "backend_list", ())
print("TorchNeuron import:", torch_neuronx.__version__)
print("PyTorch:", torch.__version__)
print("Native device: neuron")
print("Distributed backend: neuron")
PY

python3 - <<'PY'
import datasets
import safetensors
import tokenizers
import torchdata
import transformers
import wandb

print("TorchTitan Python dependencies are available.")
PY

echo
echo "TorchTitan is ready at ${TORCHTITAN_DIR}."
echo "This script does not install packages. If the dependency check failed, install"
echo "the pinned checkout's requirements inside the TorchNeuron beta environment:"
echo "  cd ${TORCHTITAN_DIR} && uv pip install --system -r requirements.txt wandb"
