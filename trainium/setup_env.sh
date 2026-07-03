#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One-time environment setup on an AWS Trainium instance (trn1 / trn2)
# running the "Deep Learning AMI Neuron (Ubuntu 22.04)".
#
# The pipeline uses TWO python environments, both derived from the venvs
# that ship with the Neuron DLAMI:
#
#   INFER_VENV  (NxD Inference + vLLM)   -> steps 0, 1, 3, 4 (generation & teacher scoring)
#   TRAIN_VENV  (torch-neuronx training) -> steps 2, 5       (SFT & Lightning OPD training)
#
# Check the exact venv names on your AMI with:  ls /opt | grep venv
# and override INFER_VENV / TRAIN_VENV below if they differ.

set -euo pipefail

INFER_VENV="${INFER_VENV:-/opt/aws_neuronx_venv_pytorch_2_7_nxd_inference}"
TRAIN_VENV="${TRAIN_VENV:-/opt/aws_neuronx_venv_pytorch_2_7}"

echo "=== [1/2] Inference env: ${INFER_VENV} ==="
# shellcheck disable=SC1091
source "${INFER_VENV}/bin/activate"
pip install --upgrade pip
pip install pandas pyarrow tqdm datasets aiohttp "huggingface_hub[cli]"

# vLLM with the AWS Neuron backend. Newer NxD-Inference DLAMIs ship it
# pre-installed; if not, install the AWS Neuron fork of vLLM.
# IMPORTANT: the branch must match your installed Neuron SDK version — see
# https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide.html
if ! python -c "import vllm" >/dev/null 2>&1; then
    echo "vLLM not found in ${INFER_VENV} — installing the AWS Neuron fork"
    VLLM_SRC=/home/ubuntu/upstreaming-to-vllm
    if [[ ! -d "${VLLM_SRC}" ]]; then
        git clone https://github.com/aws-neuron/upstreaming-to-vllm.git "${VLLM_SRC}"
    fi
    # Pick the release branch matching `dpkg -l | grep aws-neuronx-runtime` / your SDK,
    # e.g. `git -C ${VLLM_SRC} checkout neuron-2.26-vllm-v0.9`
    ( cd "${VLLM_SRC}" && VLLM_TARGET_DEVICE="neuron" pip install -e . )
fi
python -c "import vllm; print('vLLM OK:', vllm.__version__)"
deactivate

echo "=== [2/2] Training env: ${TRAIN_VENV} ==="
# shellcheck disable=SC1091
source "${TRAIN_VENV}/bin/activate"
pip install --upgrade pip
# optimum-neuron >= 0.3 is required for Qwen3 training support on Neuron.
pip install "optimum-neuron>=0.3.0" datasets pandas pyarrow tqdm
python - <<'PY'
import optimum.neuron, torch, torch_neuronx
print("optimum-neuron OK:", optimum.neuron.__version__)
PY
deactivate

echo
echo "Setup complete."
echo "  Inference venv: source ${INFER_VENV}/bin/activate"
echo "  Training venv:  source ${TRAIN_VENV}/bin/activate"
echo "Verify devices with: neuron-ls"
