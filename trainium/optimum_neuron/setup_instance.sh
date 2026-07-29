#!/usr/bin/env bash
# Add the audited Optimum Neuron Qwen3 Python layer to the DLAMI PyTorch 2.9 venv.

set -euo pipefail

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PJRT_DEVICE=NEURON
export NEURON_LOGICAL_NC_CONFIG=2

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPTIMUM_NEURON_REF="4a80f2f3de15e83e978a6f3c0d43224626d921ca"
OPTIMUM_NEURON_SRC="${OPTIMUM_NEURON_SRC:-${HOME}/optimum-neuron-qwen3-pt29}"

if [[ -z "${VENV_DIR:-}" ]]; then
    if [[ "${VIRTUAL_ENV:-}" == *"/aws_neuronx_venv_pytorch_2_9" ]] && \
       [[ -x "${VIRTUAL_ENV}/bin/python" ]]; then
        VENV_DIR="${VIRTUAL_ENV}"
    fi
fi
if [[ -z "${VENV_DIR:-}" ]]; then
    for candidate in \
        "${HOME}/aws_neuronx_venv_pytorch_2_9" \
        "/opt/aws_neuronx_venv_pytorch_2_9"; do
        if [[ -x "${candidate}/bin/python" ]]; then
            VENV_DIR="${candidate}"
            break
        fi
    done
fi

if ! command -v neuron-ls >/dev/null 2>&1; then
    echo "ERROR: neuron-ls is unavailable." >&2
    echo "Use an AWS Neuron PyTorch DLAMI on the trn2.48xlarge instance." >&2
    exit 1
fi
if ! neuron-ls >/dev/null; then
    echo "ERROR: Neuron devices are not visible to the instance." >&2
    exit 1
fi
if [[ -z "${VENV_DIR:-}" || ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "ERROR: aws_neuronx_venv_pytorch_2_9 was not found." >&2
    echo "Set VENV_DIR to its absolute path if the DLAMI stores it elsewhere." >&2
    exit 1
fi

PYTHON_BIN="${VENV_DIR}/bin/python"
echo "Using Neuron venv: ${VENV_DIR}"
echo "Using Python:      $("${PYTHON_BIN}" --version 2>&1)"

NEURON_STACK_BEFORE="$("${PYTHON_BIN}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from packaging.version import Version

required = {
    "torch": (2, 9),
    "torch-xla": (2, 9),
    "torch-neuronx": (2, 9),
    "neuronx-distributed": (0, 18),
}
bundle = [
    "torch",
    "torch-xla",
    "torch-neuronx",
    "neuronx-cc",
    "neuronx-distributed",
    "libneuronxla",
]
versions = {}
for package in bundle:
    try:
        versions[package] = version(package)
    except PackageNotFoundError as error:
        raise SystemExit(f"{package} is missing from the selected Neuron venv") from error

for package, expected in required.items():
    actual = versions[package]
    if Version(actual).release[:2] != expected:
        raise SystemExit(
            f"{package} {actual} is incompatible with the audited PyTorch 2.9 stack"
        )
for package in bundle:
    print(f"{package}=={versions[package]}")
PY
)"
printf '%s\n' "${NEURON_STACK_BEFORE}"

"${PYTHON_BIN}" -m pip install \
    -r "${SCRIPT_DIR}/requirements.txt"
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
    echo "ERROR: dependency installation changed the AWS Neuron binary bundle." >&2
    echo "Before:" >&2
    printf '%s\n' "${NEURON_STACK_BEFORE}" >&2
    echo "After:" >&2
    printf '%s\n' "${NEURON_STACK_AFTER}" >&2
    exit 1
fi

if [[ ! -d "${OPTIMUM_NEURON_SRC}/.git" ]]; then
    git clone https://github.com/huggingface/optimum-neuron.git \
        "${OPTIMUM_NEURON_SRC}"
fi
if [[ -n "$(git -C "${OPTIMUM_NEURON_SRC}" status --porcelain)" ]]; then
    echo "ERROR: ${OPTIMUM_NEURON_SRC} has local changes; refusing to alter it." >&2
    exit 1
fi
git -C "${OPTIMUM_NEURON_SRC}" fetch origin "${OPTIMUM_NEURON_REF}"
git -C "${OPTIMUM_NEURON_SRC}" checkout --detach "${OPTIMUM_NEURON_REF}"

cd "${HOME}"
PYTHONPATH="${OPTIMUM_NEURON_SRC}" "${PYTHON_BIN}" - <<'PY'
from importlib.metadata import version
import pathlib

import torch
import torch_neuronx
import torch_xla
import torch_xla.runtime as xr
import optimum.neuron
import neuronx_distributed
from optimum.neuron import NeuronTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM

module_path = pathlib.Path(torch_neuronx.__file__).resolve()
if pathlib.Path.home().joinpath("torch-neuronx") in module_path.parents:
    raise RuntimeError(
        f"torch_neuronx is being imported from a source checkout: {module_path}"
    )

required_training_arguments = {
    "tensor_parallel_size",
    "disable_sequence_parallel",
    "zero_1",
    "stochastic_rounding_enabled",
    "async_save",
}
missing_arguments = required_training_arguments - set(
    NeuronTrainingArguments.__dataclass_fields__
)
if missing_arguments:
    raise RuntimeError(
        f"Optimum Neuron training API is incompatible: missing {sorted(missing_arguments)}"
    )

print("torch:", torch.__version__)
print("torch-neuronx:", torch_neuronx.__version__)
print("torch-neuronx path:", module_path)
print("torch-xla:", version("torch-xla"))
print("neuronx-distributed:", version("neuronx-distributed"))
print("XLA device:", torch_xla.device())
print("PJRT device type:", xr.device_type())
device_count = xr.global_runtime_device_count()
print("runtime device count:", device_count)
if device_count != 64:
    raise RuntimeError(
        f"expected 64 logical NeuronCores on trn2.48xlarge LNC=2, found {device_count}"
    )
print("optimum-neuron:", optimum.neuron.__version__)
print("optimum-neuron source:", pathlib.Path(optimum.neuron.__file__).resolve())
print("trainer:", NeuronTrainer)
print("training model:", NeuronModelForCausalLM)
print("PYTORCH_2_9_NEURON_SETUP_OK")
PY

for executable in neuron_parallel_compile optimum-cli torchrun; do
    if [[ ! -x "${VENV_DIR}/bin/${executable}" ]]; then
        echo "ERROR: ${executable} is missing from ${VENV_DIR}/bin." >&2
        exit 1
    fi
done
PYTHONPATH="${OPTIMUM_NEURON_SRC}" \
    "${VENV_DIR}/bin/optimum-cli" neuron consolidate --help >/dev/null

echo
echo "Setup complete. Activate the DLAMI environment with:"
echo "source \"${VENV_DIR}/bin/activate\""
echo "The training launcher will load Optimum Neuron source from:"
echo "  ${OPTIMUM_NEURON_SRC}"
