#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Step 1 (Trainium): generate SFT data with the teacher model on Neuron.
# Mirrors scripts/generate_sft_data.sh.
#
# Required:
#   TEACHER_MODEL - HF model name or path (e.g. Qwen/Qwen3-8B)
#   SFT_PROMPTS   - prompt dataset (.jsonl or .parquet)
#   OUTPUT_DIR    - output directory
# Optional:
#   NUM_CORES     - logical NeuronCores (default: 4 = one trn2.3xlarge chip;
#                   use 32 for trn1.32xlarge, 64 for trn2.48xlarge)
#   TP_SIZE       - NeuronCores per vLLM worker (default: 4)
#
# Extra args pass through to trainium/pipeline_neuron.py, e.g. --num-samples 10

set -euo pipefail

: "${TEACHER_MODEL:?Set TEACHER_MODEL (e.g. Qwen/Qwen3-8B)}"
: "${SFT_PROMPTS:?Set SFT_PROMPTS to the prompt dataset path}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for generated SFT data}"

SFT_PROMPTS="$(cd "$(dirname "${SFT_PROMPTS}")" && pwd)/$(basename "${SFT_PROMPTS}")"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"

NUM_CORES="${NUM_CORES:-4}"
TP_SIZE="${TP_SIZE:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_curation_neuron.sh" \
    --model "${TEACHER_MODEL}" \
    --input "${SFT_PROMPTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-cores "${NUM_CORES}" \
    --tensor-parallel-size "${TP_SIZE}" \
    "$@"
