#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Build a disposable DSpark compatibility experiment on a Neuron DLAMI.
# Nothing is installed into the AWS-managed /opt environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_VENV="${INFER_VENV:-/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference}"
EXPERIMENT_VENV="${EXPERIMENT_VENV:-${SCRIPT_DIR}/.venv}"
VLLM_SRC="${VLLM_SRC:-${SCRIPT_DIR}/vendor/vllm}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_REF="${VLLM_REF:-main}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: this experiment must be built on the AWS Neuron Linux instance." >&2
    exit 1
fi

for command_name in git neuron-ls; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "ERROR: required command not found: ${command_name}" >&2
        exit 1
    fi
done

if [[ ! -x "${INFER_VENV}/bin/python" ]]; then
    echo "ERROR: NxD-Inference Python not found: ${INFER_VENV}/bin/python" >&2
    exit 1
fi

echo "=== DSpark / Trainium isolated experiment ==="
echo "Base Neuron env: ${INFER_VENV}"
echo "Experiment env:  ${EXPERIMENT_VENV}"
echo "Upstream vLLM:   ${VLLM_REPO} @ ${VLLM_REF}"
neuron-ls

# The overlay inherits the tested Neuron torch/NxDI/vllm-neuron packages. We do
# not pip-install upstream vLLM because its normal wheel/build path targets CUDA.
if [[ ! -x "${EXPERIMENT_VENV}/bin/python" ]]; then
    "${INFER_VENV}/bin/python" -m venv --system-site-packages "${EXPERIMENT_VENV}"
fi

mkdir -p "$(dirname "${VLLM_SRC}")"
if [[ ! -d "${VLLM_SRC}/.git" ]]; then
    git clone --filter=blob:none "${VLLM_REPO}" "${VLLM_SRC}"
fi
git -C "${VLLM_SRC}" fetch --depth 1 origin "${VLLM_REF}"
git -C "${VLLM_SRC}" checkout --detach FETCH_HEAD

VLLM_COMMIT="$(git -C "${VLLM_SRC}" rev-parse HEAD)"
cat > "${SCRIPT_DIR}/runtime.env" <<EOF
INFER_VENV=${INFER_VENV}
EXPERIMENT_VENV=${EXPERIMENT_VENV}
VLLM_SRC=${VLLM_SRC}
VLLM_REF=${VLLM_REF}
VLLM_COMMIT=${VLLM_COMMIT}
EOF

echo
echo "Experiment ready at vLLM commit ${VLLM_COMMIT}."
echo "Run the non-compiling compatibility probe:"
echo "  bash trainium/dspark/run_dspark_probe.sh"
echo "To attempt engine construction/Neuron compilation afterwards:"
echo "  ATTEMPT_ENGINE=1 bash trainium/dspark/run_dspark_probe.sh"

