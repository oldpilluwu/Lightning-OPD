#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Step 4 (Trainium): tokenize rollouts (Phase 1, CPU — reuses the original
# script unchanged) then precompute teacher logprobs on Neuron (Phase 2).
# Mirrors scripts/precompute_teacher_logprobs_{4b,8b}.sh.
#
# Required:
#   TEACHER_MODEL   - e.g. Qwen/Qwen3-8B (4B scale) or Qwen/Qwen3-32B (8B scale)
#   SFT_CHECKPOINT  - SFT checkpoint path (tokenizer source)
#   ROLLOUT_PARQUET - merged student rollout parquet from step 3
#   OUTPUT_DIR      - output directory (e.g. data/lightning_opd)
# Optional:
#   TP_SIZE         - NeuronCores for the teacher engine (default: 4 = one
#                     trn2.3xlarge chip; use 8 on trn1.32xlarge)
#   MAX_MODEL_LEN   - teacher context (default: 8192, as in serve_teacher_*.sh).
#                     Must exceed the longest prompt+response by >=1 token
#                     (vLLM needs 1 output token to score). The scorer checks
#                     this up front and tells you the value to use if too small.
#
# Note (8B scale): the Qwen3-32B teacher is ~64 GiB in bf16; on one 96 GiB
# chip it fits but leaves little room for KV cache. If Phase 2 OOMs, lower
# --max-num-seqs (in step4_teacher_logprobs_neuron.py) and/or MAX_MODEL_LEN.

set -euo pipefail

: "${TEACHER_MODEL:?Set TEACHER_MODEL (e.g. Qwen/Qwen3-8B)}"
: "${SFT_CHECKPOINT:?Set SFT_CHECKPOINT to the SFT model path}"
: "${ROLLOUT_PARQUET:?Set ROLLOUT_PARQUET to the student rollout parquet}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for the output parquet}"

TP_SIZE="${TP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STEM="$(basename "${ROLLOUT_PARQUET}" .parquet)"
INTERMEDIATE="${OUTPUT_DIR}/${STEM}-lightning-opd.parquet"
FINAL="${OUTPUT_DIR}/${STEM}-lightning-opd-precomputed.parquet"

# Phase 1: tokenize (CPU only) — original script, unchanged
if [[ ! -f "${INTERMEDIATE}" ]]; then
    python3 "${REPO_ROOT}/data_curation/prepare_lightning_opd.py" \
        --tokenizer-path "${SFT_CHECKPOINT}" \
        --input-parquet "${ROLLOUT_PARQUET}" \
        --output-dir "${OUTPUT_DIR}"
fi

# Phase 2: teacher logprobs on Neuron (replaces the sglang server)
python3 "${SCRIPT_DIR}/step4_teacher_logprobs_neuron.py" \
    --teacher-model "${TEACHER_MODEL}" \
    --tokenizer-path "${SFT_CHECKPOINT}" \
    --intermediate-parquet "${INTERMEDIATE}" \
    --output-parquet "${FINAL}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}"
