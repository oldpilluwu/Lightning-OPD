#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Generate SFT teacher responses using native PyTorch on Trainium (no vLLM).

set -euo pipefail

: "${TEACHER_MODEL:?Set TEACHER_MODEL (paper 4B path: Qwen/Qwen3-8B)}"
: "${SFT_PROMPTS:?Set SFT_PROMPTS to the prompt dataset path}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for generated Arrow shards}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SFT_PROMPTS="$(cd "$(dirname "${SFT_PROMPTS}")" && pwd)/$(basename "${SFT_PROMPTS}")"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"
NUM_CORES="${NUM_CORES:-1}"
TP_SIZE="${TP_SIZE:-1}"

if [[ "${TP_SIZE}" != "1" ]]; then
    echo "ERROR: native TorchNeuron curation uses one model replica per logical core." >&2
    echo "Set TP_SIZE=1; NUM_CORES controls data-parallel workers." >&2
    exit 1
fi

cd "${REPO_ROOT}"
exec bash trainium/sft_data_generation_native/run_curation_native.sh \
    --model "${TEACHER_MODEL}" \
    --input "${SFT_PROMPTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-cores "${NUM_CORES}" \
    --tensor-parallel-size 1 \
    "$@"
