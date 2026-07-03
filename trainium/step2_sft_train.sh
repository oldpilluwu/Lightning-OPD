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
#   NUM_CORES    - NeuronCores / torchrun procs (default: 32)
#   TP_SIZE      - tensor parallel size (default: 8)
#   GBS          - global batch size in packed blocks (default: 256; use 128 for 8B)
#   CUTOFF_LEN   - packed sequence length (default: 16384; lower it if you
#                  hit device OOM — 8192 halves activation memory)
#   MAX_STEPS    - default 3000
#   OPD_PROBE    - probe parquet from build_opd_probe.py; enables per-step
#                  student-NLL / fwd-KL / drift metrics on the OPD domain
#   PROBE_EVERY  - probe every N steps (default 5)
#   PROBE_SIZE   - probe sequences (default 64)

set -euo pipefail

: "${MODEL_ID:?Set MODEL_ID (e.g. Qwen/Qwen3-4B-Base)}"
: "${SFT_PARQUET:?Set SFT_PARQUET to the merged SFT parquet}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for SFT checkpoint output}"

NUM_CORES="${NUM_CORES:-32}"
TP_SIZE="${TP_SIZE:-8}"
GBS="${GBS:-256}"
CUTOFF_LEN="${CUTOFF_LEN:-16384}"
MAX_STEPS="${MAX_STEPS:-3000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Neuron compiler flags recommended for decoder-LM training
export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training ${NEURON_CC_FLAGS:-}"
export NEURON_FUSE_SOFTMAX=1
export MALLOC_ARENA_MAX=64
export XLA_DOWNCAST_BF16=0

PROBE_ARGS=()
if [[ -n "${OPD_PROBE:-}" ]]; then
    PROBE_ARGS=(
        --opd-probe-parquet "${OPD_PROBE}"
        --probe-every "${PROBE_EVERY:-5}"
        --probe-size "${PROBE_SIZE:-64}"
    )
fi

torchrun --nproc_per_node="${NUM_CORES}" \
    "${SCRIPT_DIR}/step2_sft_train_neuron.py" \
    --model-id "${MODEL_ID}" \
    --dataset-parquet "${SFT_PARQUET}" \
    --output-dir "${OUTPUT_DIR}" \
    --cutoff-len "${CUTOFF_LEN}" \
    --max-steps "${MAX_STEPS}" \
    --global-batch-size "${GBS}" \
    --tensor-parallel-size "${TP_SIZE}" \
    "${PROBE_ARGS[@]}" \
    "$@"
