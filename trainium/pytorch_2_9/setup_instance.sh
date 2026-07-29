#!/usr/bin/env bash
# Prepare the AWS DLAMI PyTorch 2.9/XLA environment without Optimum.

set -euo pipefail

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PJRT_DEVICE=NEURON
export NEURON_LOGICAL_NC_CONFIG=2

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${VENV_DIR:-}" ]]; then
    if [[ "${VIRTUAL_ENV:-}" == *"/aws_neuronx_venv_pytorch_2_9" ]] && \
       [[ -x "${VIRTUAL_ENV}/bin/python" ]]; then
        VENV_DIR="${VIRTUAL_ENV}"
    elif [[ -x "/opt/aws_neuronx_venv_pytorch_2_9/bin/python" ]]; then
        VENV_DIR="/opt/aws_neuronx_venv_pytorch_2_9"
    elif [[ -x "${HOME}/aws_neuronx_venv_pytorch_2_9/bin/python" ]]; then
        VENV_DIR="${HOME}/aws_neuronx_venv_pytorch_2_9"
    fi
fi

if [[ -z "${VENV_DIR:-}" || ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "ERROR: aws_neuronx_venv_pytorch_2_9 was not found." >&2
    exit 1
fi
if ! neuron-ls >/dev/null; then
    echo "ERROR: Neuron devices are not visible." >&2
    exit 1
fi

PYTHON_BIN="${VENV_DIR}/bin/python"
echo "Using Neuron venv: ${VENV_DIR}"
echo "Using Python:      $("${PYTHON_BIN}" --version 2>&1)"

NEURON_STACK_BEFORE="$("${PYTHON_BIN}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version

packages = [
    "torch",
    "torch-xla",
    "torch-neuronx",
    "neuronx-cc",
    "neuronx-distributed",
    "libneuronxla",
]
versions = {}
for package in packages:
    try:
        versions[package] = version(package)
    except PackageNotFoundError as error:
        raise SystemExit(f"{package} is missing from the PyTorch 2.9 venv") from error
for package in ("torch", "torch-xla", "torch-neuronx"):
    if Version(versions[package]).release[:2] != (2, 9):
        raise SystemExit(f"{package} {versions[package]} is not the PyTorch 2.9 stack")
nxd = Version(versions["neuronx-distributed"])
if not (Version("0.19") <= nxd < Version("0.20")):
    raise SystemExit(f"neuronx-distributed {nxd} is not the required 0.19 line")
for package in packages:
    print(f"{package}=={versions[package]}")
PY
)"
printf '%s\n' "${NEURON_STACK_BEFORE}"

OLD_OPTIMUM_SOURCE="${HOME}/optimum-neuron-qwen3-pt29"
if [[ -e "${OLD_OPTIMUM_SOURCE}" ]]; then
    if [[ "$(realpath -- "${OLD_OPTIMUM_SOURCE}")" != \
          "$(realpath -- "${HOME}")/optimum-neuron-qwen3-pt29" ]]; then
        echo "ERROR: refusing to remove unexpected path: ${OLD_OPTIMUM_SOURCE}" >&2
        exit 1
    fi
    rm -rf -- "${OLD_OPTIMUM_SOURCE}"
    echo "Removed obsolete Optimum source checkout: ${OLD_OPTIMUM_SOURCE}"
fi
OLD_REPO_IMPLEMENTATION="${REPO_DIR}/trainium/optimum_neuron"
if [[ -d "${OLD_REPO_IMPLEMENTATION}" ]]; then
    if [[ "$(realpath -- "${OLD_REPO_IMPLEMENTATION}")" != \
          "${REPO_DIR}/trainium/optimum_neuron" ]]; then
        echo "ERROR: refusing to remove unexpected path: ${OLD_REPO_IMPLEMENTATION}" >&2
        exit 1
    fi
    rm -rf -- "${OLD_REPO_IMPLEMENTATION}"
    echo "Removed obsolete repository cache: ${OLD_REPO_IMPLEMENTATION}"
fi
"${PYTHON_BIN}" -m pip uninstall -y optimum optimum-neuron || true
"${PYTHON_BIN}" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
"${PYTHON_BIN}" -m pip install \
    --force-reinstall \
    --no-deps \
    "transformers==4.57.1"
"${PYTHON_BIN}" -m pip check

NEURON_STACK_AFTER="$("${PYTHON_BIN}" - <<'PY'
from importlib.metadata import version
for package in [
    "torch",
    "torch-xla",
    "torch-neuronx",
    "neuronx-cc",
    "neuronx-distributed",
    "libneuronxla",
]:
    print(f"{package}=={version(package)}")
PY
)"
if [[ "${NEURON_STACK_AFTER}" != "${NEURON_STACK_BEFORE}" ]]; then
    echo "ERROR: Python dependency setup changed the AWS Neuron binary bundle." >&2
    printf 'Before:\n%s\nAfter:\n%s\n' \
        "${NEURON_STACK_BEFORE}" "${NEURON_STACK_AFTER}" >&2
    exit 1
fi

cd "${HOME}"
PYTHONPATH="${SCRIPT_DIR}" "${PYTHON_BIN}" - <<'PY'
from importlib.metadata import version
import torch
import torch_neuronx
import torch_xla
import torch_xla.runtime as xr
import neuronx_distributed as nxd
from neuronx_distributed.kernels.flash_attn import nki_flash_attn_func
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from transformers import AutoTokenizer, Qwen3Config
from transformers.generation import GenerationMixin
from modeling_qwen3_nxd import Qwen3ForCausalLM

print("torch:", torch.__version__)
print("torch-neuronx:", version("torch-neuronx"))
print("torch-xla:", version("torch-xla"))
print("neuronx-distributed:", version("neuronx-distributed"))
print("XLA device:", torch_xla.device())
print("PJRT device type:", xr.device_type())
print("runtime device count:", xr.global_runtime_device_count())
if xr.global_runtime_device_count() != 64:
    raise RuntimeError("Expected 64 logical NeuronCores on trn2.48xlarge LNC=2")
print("Qwen3 config:", Qwen3Config)
print("Qwen3 direct model:", Qwen3ForCausalLM)
print("NxD layers:", ParallelEmbedding, ColumnParallelLinear, RowParallelLinear)
print("NxD flash attention:", nki_flash_attn_func)
print("Tokenizer:", AutoTokenizer)
print("GenerationMixin:", GenerationMixin)
print("PYTORCH_2_9_DIRECT_SETUP_OK")
PY

for executable in torchrun neuron_parallel_compile; do
    if [[ ! -x "${VENV_DIR}/bin/${executable}" ]]; then
        echo "ERROR: ${executable} is missing from ${VENV_DIR}/bin." >&2
        exit 1
    fi
done

echo "Setup complete."
