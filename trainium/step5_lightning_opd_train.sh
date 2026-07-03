#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Step 5 (Trainium): Lightning OPD training on all NeuronCores.
# Mirrors configs/lightning_opd/qwen3-{4b,8b}-lightning-opd.py.
#
# Required:
#   SFT_CHECKPOINT     - consolidated HF checkpoint from step 2
#   LIGHTNING_OPD_DATA - precomputed parquet from step 4
#   OUTPUT_DIR         - checkpoint output directory
# Optional:
#   NUM_CORES   - torchrun procs (default: 32)
#   TP_SIZE     - tensor parallel size (default: 8; original used TP=2 for 4B
#                 and TP=4 for 8B on H100s — Trainium cores have less memory,
#                 hence the higher default)
#   MAX_STEPS   - default 150 (README: ~150 steps sufficient for convergence;
#                 the original config caps at 3000)
#   MAX_SEQ_LEN - static padded length (default 5632 = 4096 response + prompt)
#   OPD_PROBE   - probe parquet from build_opd_probe.py (same one used in
#                 step 2) to keep tracking fwd-KL/drift during OPD
#   PROBE_EVERY / PROBE_SIZE - probe cadence / size (defaults 5 / 64)

set -euo pipefail

: "${SFT_CHECKPOINT:?Set SFT_CHECKPOINT to the consolidated SFT checkpoint}"
: "${LIGHTNING_OPD_DATA:?Set LIGHTNING_OPD_DATA to the precomputed parquet}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for OPD checkpoint output}"

NUM_CORES="${NUM_CORES:-32}"
TP_SIZE="${TP_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-150}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-5632}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    "${SCRIPT_DIR}/step5_lightning_opd_train_neuron.py" \
    --sft-checkpoint "${SFT_CHECKPOINT}" \
    --data-parquet "${LIGHTNING_OPD_DATA}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-steps "${MAX_STEPS}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --tensor-parallel-size "${TP_SIZE}" \
    "${PROBE_ARGS[@]}" \
    "$@"
