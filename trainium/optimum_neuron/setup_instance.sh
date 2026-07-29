#!/usr/bin/env bash
# Install the pinned Optimum Neuron Qwen3 training stack on a Neuron DLAMI.

set -euo pipefail

unset PYTHONPATH
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${HOME}/venvs/lightning-opd-sft}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
NEURON_INDEX_URL="${NEURON_INDEX_URL:-https://pip.repos.neuron.amazonaws.com}"

if ! command -v neuron-ls >/dev/null 2>&1; then
    echo "ERROR: neuron-ls is unavailable." >&2
    echo "Use an AWS Neuron PyTorch DLAMI on the trn2.48xlarge instance." >&2
    exit 1
fi
if ! neuron-ls >/dev/null; then
    echo "ERROR: Neuron devices are not visible to the instance." >&2
    exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON_BIN} is unavailable." >&2
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install \
    --extra-index-url "${NEURON_INDEX_URL}" \
    -r "${SCRIPT_DIR}/requirements.txt"
"${VENV_DIR}/bin/python" -m pip check

cd "${HOME}"
"${VENV_DIR}/bin/python" - <<'PY'
import pathlib

import torch
import torch_neuronx
import optimum.neuron
from optimum.neuron import NeuronTrainer
from optimum.neuron.models.training import NeuronModelForCausalLM

module_path = pathlib.Path(torch_neuronx.__file__).resolve()
if pathlib.Path.home().joinpath("torch-neuronx") in module_path.parents:
    raise RuntimeError(
        f"torch_neuronx is being imported from a source checkout: {module_path}"
    )

print("torch:", torch.__version__)
print("torch-neuronx:", torch_neuronx.__version__)
print("torch-neuronx path:", module_path)
print("optimum-neuron:", optimum.neuron.__version__)
print("trainer:", NeuronTrainer)
print("training model:", NeuronModelForCausalLM)
print("NEURON_SETUP_OK")
PY

echo
echo "Setup complete. Activate it with:"
echo "source \"${VENV_DIR}/bin/activate\""
