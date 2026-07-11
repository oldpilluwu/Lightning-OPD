#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Validate the already-installed vLLM-Neuron environment for the DSpark probe.
# This script intentionally does not clone or install vLLM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_VENV="${INFER_VENV:-/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: run this experiment on the AWS Neuron Linux instance." >&2
    exit 1
fi

if ! command -v neuron-ls >/dev/null 2>&1; then
    echo "ERROR: neuron-ls is unavailable; use a Neuron DLAMI/instance." >&2
    exit 1
fi

PYTHON="${INFER_VENV}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: Neuron inference Python not found: ${PYTHON}" >&2
    exit 1
fi

# Invoking the venv's Python by absolute path does not activate the venv.
# torch-neuronx starts helper programs such as libneuronpjrt-path by name, so
# the venv's bin directory must also be present on PATH.
export PATH="${INFER_VENV}/bin:${PATH}"

if ! command -v libneuronpjrt-path >/dev/null 2>&1; then
    echo "ERROR: libneuronpjrt-path is not available in ${INFER_VENV}/bin." >&2
    echo "The selected environment is incomplete or is not a torch-neuronx environment." >&2
    exit 1
fi

echo "=== Installed vLLM-Neuron DSpark experiment ==="
echo "Neuron env: ${INFER_VENV}"
neuron-ls

"${PYTHON}" - <<'PY'
from importlib import metadata

for package in (
    "torch-neuronx",
    "neuronx-distributed",
    "neuronx-distributed-inference",
    "vllm",
    "vllm-neuron",
):
    print(f"{package}={metadata.version(package)}")

import neuronx_distributed_inference
import torch_neuronx
import vllm
import vllm_neuron

print("Installed Neuron imports: PASS")
PY

cat > "${SCRIPT_DIR}/runtime.env" <<EOF
INFER_VENV=${INFER_VENV}
EOF

echo
echo "No vLLM source checkout was created and no package was installed."
echo "Run the non-compiling compatibility probe:"
echo "  bash trainium/dspark/run_dspark_probe.sh"
echo "Attempt the real DSpark configuration (downloads both model checkpoints):"
echo "  ATTEMPT_ENGINE=1 bash trainium/dspark/run_dspark_probe.sh"
echo "Optionally test the checkpoint as an ordinary NxDI draft model (not DSpark):"
echo "  ATTEMPT_ENGINE=1 SPECULATIVE_METHOD=draft_model bash trainium/dspark/run_dspark_probe.sh"
