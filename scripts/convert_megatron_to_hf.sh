#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convert a Megatron torch_dist checkpoint to HuggingFace format.
#
# Required env vars:
#   MEGATRON_CKPT_DIR  - path to the Megatron checkpoint directory (e.g., /root/models/<name>_ckpt__<config>/iter_0000150)
#   HF_OUTPUT_DIR      - path to save the converted HuggingFace model
#   ORIGIN_HF_DIR      - path to the original HuggingFace model (for config.json, tokenizer, etc.)
#
# Example:
#   MEGATRON_CKPT_DIR=/root/models/Qwen3-4B-Base-sft_ckpt__qwen3-4b-lightning-opd/iter_0000150 \
#   HF_OUTPUT_DIR=checkpoints/qwen3-4b-lightning-opd-hf \
#   ORIGIN_HF_DIR=checkpoints/qwen3-4b-base-sft-qwen3-8b/<your-sft-checkpoint> \
#   bash scripts/convert_megatron_to_hf.sh

set -euo pipefail

: "${MEGATRON_CKPT_DIR:?Please set MEGATRON_CKPT_DIR}"
: "${HF_OUTPUT_DIR:?Please set HF_OUTPUT_DIR}"
: "${ORIGIN_HF_DIR:?Please set ORIGIN_HF_DIR}"

python tools/convert_torch_dist_to_hf.py \
    --input-dir "${MEGATRON_CKPT_DIR}" \
    --output-dir "${HF_OUTPUT_DIR}" \
    --origin-hf-dir "${ORIGIN_HF_DIR}"
