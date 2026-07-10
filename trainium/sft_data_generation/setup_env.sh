#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One-time setup for trainium/sft_data_generation/generate_sft_data.sh.
#
# Recommended EC2 host:
#   - trn2.3xlarge
#   - Deep Learning AMI Neuron, Ubuntu 24.04
#   - PyTorch 2.9 NxD-Inference venv under /opt
#   - at least 1 TiB gp3 storage (2 TiB recommended for a full 300K run)
#
# Usage from the repository root:
#   bash trainium/sft_data_generation/setup_env.sh
#
# Optional overrides:
#   INFER_VENV=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference
#   VLLM_NEURON_REF=0.5.0
#   VLLM_NEURON_SRC=$HOME/src/vllm-neuron
#   HF_HOME=/mnt/data/huggingface
#   HF_TOKEN=hf_...  # only needed for gated/private resources

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

INFER_VENV="${INFER_VENV:-/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference}"
VLLM_NEURON_REF="${VLLM_NEURON_REF:-0.5.0}"
VLLM_NEURON_SRC="${VLLM_NEURON_SRC:-${HOME}/src/vllm-neuron}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://pip.repos.neuron.amazonaws.com}"

echo "=== Lightning-OPD SFT generation environment setup ==="
echo "Repository:       ${REPO_ROOT}"
echo "Inference venv:   ${INFER_VENV}"
echo "vllm-neuron ref:  ${VLLM_NEURON_REF}"
echo "Hugging Face dir: ${HF_HOME}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: run this setup on the AWS Neuron Linux instance." >&2
    exit 1
fi

if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    echo "Operating system: ${PRETTY_NAME:-unknown}"
    if [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" != "24.04" ]]; then
        echo "WARNING: Ubuntu 24.04 is the supported DLAMI baseline; found ${VERSION_ID:-unknown}." >&2
    fi
fi

for command_name in git python3 neuron-ls; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "ERROR: required command not found: ${command_name}" >&2
        echo "Launch an AWS Deep Learning AMI Neuron image rather than plain Ubuntu." >&2
        exit 1
    fi
done

if [[ ! -f "${INFER_VENV}/bin/activate" ]]; then
    echo "ERROR: PyTorch 2.9 NxD-Inference venv not found: ${INFER_VENV}" >&2
    echo "Available Neuron environments:" >&2
    find /opt -maxdepth 1 -type d -name 'aws_neuronx_venv_*' -print >&2 || true
    exit 1
fi

echo "Neuron devices:"
neuron-ls

AVAILABLE_KIB="$(df -Pk "${REPO_ROOT}" | awk 'NR==2 {print $4}')"
AVAILABLE_GIB=$(( AVAILABLE_KIB / 1024 / 1024 ))
echo "Free repository-volume space: ${AVAILABLE_GIB} GiB"
if (( AVAILABLE_GIB < 750 )); then
    echo "WARNING: less than 750 GiB is free. The full 300K/16K-token generation" >&2
    echo "         can require substantial checkpoint, cache, and dataset storage." >&2
fi

# shellcheck disable=SC1091
source "${INFER_VENV}/bin/activate"

python - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(f"Python >=3.10 is required; found {sys.version}")
print("Python:", sys.version.replace("\n", " "))
PY

python -m pip install --upgrade pip
python -m pip install \
    datasets \
    pandas \
    pyarrow \
    tqdm \
    aiohttp \
    "huggingface_hub[cli]"

mkdir -p "$(dirname "${VLLM_NEURON_SRC}")" "${HF_HOME}"

if [[ -d "${VLLM_NEURON_SRC}/.git" ]]; then
    echo ">>> Updating existing vllm-neuron checkout..."
    git -C "${VLLM_NEURON_SRC}" fetch --tags origin
else
    echo ">>> Cloning vllm-neuron..."
    git clone https://github.com/vllm-project/vllm-neuron.git "${VLLM_NEURON_SRC}"
fi

if ! git -C "${VLLM_NEURON_SRC}" rev-parse --verify --quiet \
    "refs/tags/${VLLM_NEURON_REF}" >/dev/null; then
    if ! git -C "${VLLM_NEURON_SRC}" rev-parse --verify --quiet \
        "origin/${VLLM_NEURON_REF}" >/dev/null; then
        echo "ERROR: vllm-neuron ref '${VLLM_NEURON_REF}' was not found." >&2
        echo "Set VLLM_NEURON_REF to the version matched to the installed Neuron SDK." >&2
        exit 1
    fi
    VLLM_NEURON_GIT_REF="origin/${VLLM_NEURON_REF}"
else
    VLLM_NEURON_GIT_REF="refs/tags/${VLLM_NEURON_REF}"
fi

git -C "${VLLM_NEURON_SRC}" checkout --detach "${VLLM_NEURON_GIT_REF}"
python -m pip install \
    --extra-index-url "${PIP_EXTRA_INDEX_URL}" \
    -e "${VLLM_NEURON_SRC}"

export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export HF_HOME

python - "${VLLM_NEURON_REF}" <<'PY'
import importlib.metadata as metadata
import importlib.util
import sys

expected_plugin = sys.argv[1]
required_modules = (
    "torch",
    "torch_neuronx",
    "neuronx_distributed_inference",
    "vllm",
    "vllm_neuron",
    "datasets",
    "pandas",
    "pyarrow",
)
missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing required Python modules: {missing}")

for package in (
    "torch",
    "torch-neuronx",
    "neuronx-distributed-inference",
    "vllm",
    "vllm-neuron",
):
    try:
        print(f"{package}: {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package}: installed, version metadata unavailable")

actual_plugin = metadata.version("vllm-neuron")
if actual_plugin != expected_plugin:
    raise SystemExit(
        f"Expected vllm-neuron {expected_plugin}, but installed {actual_plugin}. "
        "Use a clean compatible inference venv or choose the matching ref."
    )
PY

if [[ -n "${HF_TOKEN:-}" ]]; then
    if command -v hf >/dev/null 2>&1; then
        hf auth login --token "${HF_TOKEN}"
    else
        huggingface-cli login --token "${HF_TOKEN}"
    fi
fi

python -m pip check

cat > "${SCRIPT_DIR}/runtime.env" <<EOF
# Generated by setup_env.sh. Re-run setup_env.sh to refresh it.
export INFER_VENV="${INFER_VENV}"
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export HF_HOME="${HF_HOME}"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"
EOF

echo
echo "=== Setup complete ==="
echo "Runtime configuration: ${SCRIPT_DIR}/runtime.env"
echo
echo "Run a faithful 64-prompt smoke generation first:"
echo "  SMOKE=1 bash trainium/sft_data_generation/generate_sft_data.sh"
echo
echo "Then run the full 300K generation:"
echo "  bash trainium/sft_data_generation/generate_sft_data.sh"
