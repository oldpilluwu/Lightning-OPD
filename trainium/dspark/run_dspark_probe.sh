#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/runtime.env" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/runtime.env"
fi

INFER_VENV="${INFER_VENV:-/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference}"
EXPERIMENT_VENV="${EXPERIMENT_VENV:-${SCRIPT_DIR}/.venv}"
VLLM_SRC="${VLLM_SRC:-${SCRIPT_DIR}/vendor/vllm}"
USE_UPSTREAM_VLLM="${USE_UPSTREAM_VLLM:-1}"
ATTEMPT_ENGINE="${ATTEMPT_ENGINE:-0}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
SPECULATOR_MODEL="${SPECULATOR_MODEL:-deepseek-ai/dspark_qwen3_8b_block7}"
TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"

if [[ "${USE_UPSTREAM_VLLM}" == "1" ]]; then
    PYTHON="${EXPERIMENT_VENV}/bin/python"
    if [[ ! -x "${PYTHON}" || ! -d "${VLLM_SRC}/vllm" ]]; then
        echo "ERROR: source experiment is not set up." >&2
        echo "Run: bash trainium/dspark/setup_experiment.sh" >&2
        exit 1
    fi
    export PYTHONPATH="${VLLM_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
    STACK_LABEL="upstream vLLM source over the Neuron environment"
else
    PYTHON="${INFER_VENV}/bin/python"
    STACK_LABEL="currently installed Neuron/vLLM stack"
fi

mkdir -p "${SCRIPT_DIR}/logs"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${SCRIPT_DIR}/logs/probe-${TIMESTAMP}.log"

echo "Testing ${STACK_LABEL}"
echo "Log: ${LOG_FILE}"

ARGS=(
    --target-model "${TARGET_MODEL}"
    --speculator-model "${SPECULATOR_MODEL}"
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
    echo "EXPECTED EXPERIMENTAL FAILURE (exit ${STATUS}). See ${LOG_FILE}" >&2
fi
exit "${STATUS}"

