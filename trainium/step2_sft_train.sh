#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Step 2 (Trainium): launch SFT training on all NeuronCores of one instance.
# Mirrors configs/sft/run_sft.sh (LlamaFactory + torchrun).
#
# Required:
#   MODEL_ID     - base model (e.g. Qwen/Qwen3-4B-Base)
#   SFT_PARQUET  - merged SFT parquet from step 1
#   OUTPUT_DIR   - checkpoint output directory
# Optional:
#   NUM_CORES    - logical NeuronCores / torchrun procs (default: 4 = one
#                  trn2.3xlarge chip; use 32 for trn1.32xlarge)
#   TP_SIZE      - tensor parallel size (default: 4; must divide NUM_CORES and
#                  the model's KV-head count. DP = NUM_CORES / TP_SIZE)
#   GBS          - global batch size in packed blocks (default: 256; use 128 for 8B)
#   CUTOFF_LEN   - packed sequence length (default: 16384 = paper value). On a
#                  single 96 GiB chip this is the main OOM lever — lower to
#                  8192 (or 4096) if step 2 OOMs; changes packing, not the data)
#   MAX_STEPS    - default 3000

set -euo pipefail

: "${MODEL_ID:?Set MODEL_ID (e.g. Qwen/Qwen3-4B-Base)}"
: "${SFT_PARQUET:?Set SFT_PARQUET to the merged SFT parquet}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for SFT checkpoint output}"

NUM_CORES="${NUM_CORES:-4}"
TP_SIZE="${TP_SIZE:-4}"
GBS="${GBS:-256}"
CUTOFF_LEN="${CUTOFF_LEN:-16384}"
MAX_STEPS="${MAX_STEPS:-3000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Neuron compiler flags recommended for decoder-LM training
export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training ${NEURON_CC_FLAGS:-}"
export NEURON_FUSE_SOFTMAX=1
export MALLOC_ARENA_MAX=64
export XLA_DOWNCAST_BF16=0

torchrun --nproc_per_node="${NUM_CORES}" \
    "${SCRIPT_DIR}/step2_sft_train_neuron.py" \
    --model-id "${MODEL_ID}" \
    --dataset-parquet "${SFT_PARQUET}" \
    --output-dir "${OUTPUT_DIR}" \
    --cutoff-len "${CUTOFF_LEN}" \
    --max-steps "${MAX_STEPS}" \
    --global-batch-size "${GBS}" \
    --tensor-parallel-size "${TP_SIZE}" \
    "$@"
