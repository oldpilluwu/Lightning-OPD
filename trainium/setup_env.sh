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

# Auto-detect the Neuron venvs that ship with the DLAMI. Names carry the
# PyTorch version (e.g. _2_7, _2_9), so we match by role rather than exact name.
# Override by exporting INFER_VENV / TRAIN_VENV before running.
#
#   INFER_VENV -> a venv with vLLM (preferred) or the *_nxd_inference venv
#   TRAIN_VENV -> the plain torch-neuronx training venv (no _inference/_vllm)
pick_venv() {
    # $1 = grep -E pattern (matched against basename), prints first match under /opt
    local d
    for d in /opt/aws_neuronx_venv_*; do
        [[ -x "${d}/bin/activate" || -f "${d}/bin/activate" ]] || continue
        if basename "${d}" | grep -Eq "$1"; then
            echo "${d}"
            return 0
        fi
    done
    return 1
}

if [[ -z "${INFER_VENV:-}" ]]; then
    INFER_VENV="$(pick_venv 'vllm')" \
        || INFER_VENV="$(pick_venv 'nxd_inference|inference')" \
        || { echo "ERROR: could not find an inference venv under /opt. Set INFER_VENV manually." >&2; \
             echo "Available:" >&2; ls -d /opt/aws_neuronx_venv_* 2>/dev/null >&2; exit 1; }
fi

if [[ -z "${TRAIN_VENV:-}" ]]; then
    # A pytorch venv that is NOT an inference/vllm venv.
    TRAIN_VENV="$(pick_venv 'pytorch_[0-9]' | grep -Ev 'inference|vllm' | head -n1)"
    if [[ -z "${TRAIN_VENV}" ]]; then
        for d in /opt/aws_neuronx_venv_pytorch_*; do
            case "$(basename "${d}")" in
                *inference*|*vllm*) continue ;;
                *) TRAIN_VENV="${d}"; break ;;
            esac
        done
    fi
    [[ -n "${TRAIN_VENV}" ]] \
        || { echo "ERROR: could not find a training venv under /opt. Set TRAIN_VENV manually." >&2; \
             echo "Available:" >&2; ls -d /opt/aws_neuronx_venv_* 2>/dev/null >&2; exit 1; }
fi

echo "Detected venvs:"
echo "  INFER_VENV=${INFER_VENV}"
echo "  TRAIN_VENV=${TRAIN_VENV}"
echo

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
