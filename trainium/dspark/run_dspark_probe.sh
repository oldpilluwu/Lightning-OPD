#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/runtime.env" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/runtime.env"
fi

INFER_VENV="${INFER_VENV:-/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference}"
ATTEMPT_ENGINE="${ATTEMPT_ENGINE:-0}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
SPECULATOR_MODEL="${SPECULATOR_MODEL:-deepseek-ai/dspark_qwen3_8b_block7}"
SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-dspark}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"

PYTHON="${INFER_VENV}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: installed Neuron inference Python not found: ${PYTHON}" >&2
    echo "Run: bash trainium/dspark/setup_experiment.sh" >&2
    exit 1
fi

# Select the installed vLLM hardware plugin and keep V1 in-process. The latter
# gives useful Python errors for this experimental path and avoids a second
# engine process owning the same NeuronCores.
export VLLM_NEURON_FRAMEWORK="${VLLM_NEURON_FRAMEWORK:-neuronx-distributed-inference}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

mkdir -p "${SCRIPT_DIR}/logs"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${SCRIPT_DIR}/logs/probe-${TIMESTAMP}.log"

echo "Testing the installed vLLM-Neuron stack"
echo "Speculative method: ${SPECULATIVE_METHOD}"
echo "Target model:       ${TARGET_MODEL}"
echo "Speculator model:   ${SPECULATOR_MODEL}"
echo "Log:                ${LOG_FILE}"

ARGS=(
    --target-model "${TARGET_MODEL}"
    --speculator-model "${SPECULATOR_MODEL}"
    --speculative-method "${SPECULATIVE_METHOD}"
    --tensor-parallel-size "${TP_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}"
)
if [[ "${ATTEMPT_ENGINE}" == "1" ]]; then
    ARGS+=(--attempt-engine)
fi

set +e
"${PYTHON}" "${SCRIPT_DIR}/probe_dspark.py" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
STATUS="${PIPESTATUS[0]}"
set -e

if (( STATUS == 0 )); then
    echo "PASS: the requested probe completed. See ${LOG_FILE}"
else
    echo "EXPERIMENT RESULT (exit ${STATUS}). See ${LOG_FILE}" >&2
fi
exit "${STATUS}"
