#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Step 1: Generate SFT data using the teacher model.
#
# Uses data_curation/ to run the teacher model on OpenThoughts-3 prompts
# and generate response trajectories for SFT training.
#
# Required environment variables:
#   TEACHER_MODEL - HuggingFace model name or path (e.g. Qwen/Qwen3-8B)
#   SFT_PROMPTS   - Path to the prompt dataset (.jsonl or .parquet)
#   OUTPUT_DIR    - Directory for generated SFT data
#
# Optional:
#   NUM_GPUS      - Number of GPUs to use (default: 8)
#   TP_SIZE       - Tensor parallel size per worker (default: 1)
#
# Extra args are passed through to data_curation/pipeline.py, e.g.:
#   bash scripts/generate_sft_data.sh --num-samples 10

set -euo pipefail

: "${TEACHER_MODEL:?Set TEACHER_MODEL (e.g. Qwen/Qwen3-8B)}"
: "${SFT_PROMPTS:?Set SFT_PROMPTS to the prompt dataset path}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for generated SFT data}"

# Resolve to absolute paths (workers may run from different cwd)
SFT_PROMPTS="$(cd "$(dirname "${SFT_PROMPTS}")" && pwd)/$(basename "${SFT_PROMPTS}")"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"

NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-1}"

bash data_curation/run_curation.sh \
    --model "${TEACHER_MODEL}" \
    --input "${SFT_PROMPTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-gpus "${NUM_GPUS}" \
    --tensor-parallel-size "${TP_SIZE}" \
    "$@"
