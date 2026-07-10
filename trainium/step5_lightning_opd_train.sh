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
#   NUM_CORES   - logical NeuronCores / torchrun procs (default: 4 = one
#                 trn2.3xlarge chip; use 32 for trn1.32xlarge)
#   TP_SIZE     - tensor parallel size (default: 4; must divide NUM_CORES.
#                 The original slime configs used TP=2 (4B) / TP=4 (8B) on
#                 H100s with data parallelism; on one Trainium2 chip there is
#                 no DP, so TP=NUM_CORES=4 is the whole device)
#   MAX_STEPS   - default 150 (README: ~150 steps sufficient for convergence;
#                 the original config caps at 3000)
#   GBS         - global batch size in sequences (default 256 = paper; the
#                 smoke run lowers this so a few dozen rollouts give real steps)
#   MAX_SEQ_LEN - static padded length (default 5632 = 4096 response + 1536
#                 prompt budget). Rows with prompt+response above this are
#                 dropped (slime keeps them) — the trainer warns and aborts if
#                 >2% drop. Raise it (costs device memory) if you see drops.

set -euo pipefail

: "${SFT_CHECKPOINT:?Set SFT_CHECKPOINT to the consolidated SFT checkpoint}"
: "${LIGHTNING_OPD_DATA:?Set LIGHTNING_OPD_DATA to the precomputed parquet}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for OPD checkpoint output}"

NUM_CORES="${NUM_CORES:-4}"
TP_SIZE="${TP_SIZE:-4}"
MAX_STEPS="${MAX_STEPS:-150}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-5632}"
GBS="${GBS:-256}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training ${NEURON_CC_FLAGS:-}"
export NEURON_FUSE_SOFTMAX=1
export MALLOC_ARENA_MAX=64
export XLA_DOWNCAST_BF16=0

torchrun --nproc_per_node="${NUM_CORES}" \
    "${SCRIPT_DIR}/step5_lightning_opd_train_neuron.py" \
    --sft-checkpoint "${SFT_CHECKPOINT}" \
    --data-parquet "${LIGHTNING_OPD_DATA}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-steps "${MAX_STEPS}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --global-batch-size "${GBS}" \
    --tensor-parallel-size "${TP_SIZE}" \
    "$@"
