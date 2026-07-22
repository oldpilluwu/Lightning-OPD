#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Environment bootstrap for the NxD Inference-backed vLLM integration.
# This is sourced by run_sft_curation_trn2_48xlarge.sh so activation and
# exported variables remain active for the complete curation run.

set -euo pipefail

NXD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NXD_REPO_ROOT="$(cd "${NXD_SETUP_DIR}/../.." && pwd)"
WORK_ROOT="${WORK_ROOT:-${NXD_REPO_ROOT}/data/trainium/nxd}"
VLLM_NEURON_REF="0.5.0"
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://pip.repos.neuron.amazonaws.com}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: this environment must be prepared on the Trainium Linux host." >&2
    exit 1
fi

for required_command in curl git neuron-ls python3; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "ERROR: missing ${required_command}. Use an AWS Neuron DLAMI." >&2
        exit 1
    fi
done

# The new Neuron 2.31 direct plugin does not yet support Qwen3ForCausalLM.
# Select the PyTorch 2.9 NxD Inference environment used by the documented
# vLLM 0.16 + vllm-neuron 0.5 integration instead.
if [[ -z "${INFER_VENV:-}" ]]; then
    VENV_CANDIDATES=(
        "${HOME}/aws_neuronx_venv_pytorch_2_9_nxd_inference"
        "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
        "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16"
    )
    for candidate in "${VENV_CANDIDATES[@]}"; do
        if [[ -f "${candidate}/bin/activate" ]]; then
            INFER_VENV="${candidate}"
            break
        fi
    done
fi

if [[ -z "${INFER_VENV:-}" || ! -f "${INFER_VENV}/bin/activate" ]]; then
    cat >&2 <<'EOF'
ERROR: a PyTorch 2.9 NxD Inference virtual environment was not found.

Use an Ubuntu 24.04 AWS Neuron DLAMI that includes NxD Inference, or set:
  INFER_VENV=/path/to/aws_neuronx_venv_pytorch_2_9_nxd_inference

Do not point INFER_VENV at the vLLM 0.21 direct-beta environment; that
environment does not provide Qwen3ForCausalLM.
EOF
    exit 1
fi

# shellcheck disable=SC1090
source "${INFER_VENV}/bin/activate"

export INFER_VENV
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export NEURON_LOGICAL_NC_CONFIG="2"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

python - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(f"Python 3.10+ is required; found {sys.version}")
print("Python:", sys.version.replace("\n", " "))
PY

echo ">>> Installing data-curation dependencies into ${INFER_VENV}"
python -m pip install \
    datasets \
    pandas \
    pyarrow \
    tqdm \
    "huggingface_hub[cli]"

if ! python - "${VLLM_NEURON_REF}" <<'PY'
import importlib.metadata as metadata
import sys

expected = sys.argv[1]
try:
    actual_plugin = metadata.version("vllm-neuron")
    actual_vllm = metadata.version("vllm")
except metadata.PackageNotFoundError:
    raise SystemExit(1)
versions_match = actual_plugin == expected and actual_vllm.startswith("0.16.0")
raise SystemExit(0 if versions_match else 1)
PY
then
    VLLM_NEURON_SRC="${VLLM_NEURON_SRC:-${WORK_ROOT}/src/vllm-neuron-${VLLM_NEURON_REF}}"
    mkdir -p "$(dirname "${VLLM_NEURON_SRC}")"

    if [[ ! -d "${VLLM_NEURON_SRC}/.git" ]]; then
        echo ">>> Cloning vllm-neuron ${VLLM_NEURON_REF}"
        git clone --depth 1 --branch "${VLLM_NEURON_REF}" \
            https://github.com/vllm-project/vllm-neuron.git \
            "${VLLM_NEURON_SRC}"
    fi

    CHECKED_OUT_VERSION="$(git -C "${VLLM_NEURON_SRC}" describe --tags --exact-match 2>/dev/null || true)"
    if [[ "${CHECKED_OUT_VERSION}" != "${VLLM_NEURON_REF}" ]]; then
        echo "ERROR: ${VLLM_NEURON_SRC} is not checked out at ${VLLM_NEURON_REF}." >&2
        echo "Use an empty VLLM_NEURON_SRC or point it at the required tag." >&2
        exit 1
    fi

    echo ">>> Installing the documented NxD vLLM plugin"
    python -m pip install \
        --extra-index-url "${PIP_EXTRA_INDEX_URL}" \
        -e "${VLLM_NEURON_SRC}"
fi

python - <<'PY'
import importlib.metadata as metadata
import importlib.util
import vllm

required_modules = (
    "torch",
    "torch_neuronx",
    "neuronx_distributed_inference",
    "vllm_neuron",
    "datasets",
    "pyarrow",
)
missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing required modules: {missing}")

print(f"vLLM: {vllm.__version__}")
for package in (
    "vllm-neuron",
    "torch",
    "torch-neuronx",
    "neuronx-distributed-inference",
):
    try:
        print(f"{package}: {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package}: version metadata unavailable")

if metadata.version("vllm-neuron") != "0.5.0":
    raise SystemExit("Expected vllm-neuron 0.5.0")
if not metadata.version("vllm").startswith("0.16.0"):
    raise SystemExit("Expected vLLM 0.16.0")
PY

python -m pip check
echo ">>> Neuron devices"
neuron-ls
