#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Launch data curation across multiple GPUs / nodes.
#
# Each GPU runs one independent vLLM worker that processes a disjoint shard
# of the dataset. No torch.distributed communication is needed — each worker
# is a standalone process with its own rank derived from environment variables.
#
# ── Single node, 8 GPUs (tp=1, 8 workers) ────────────────────────────────
#   bash data_curation/run_curation.sh \
#       --model Qwen/Qwen3-4B \
#       --input data.jsonl \
#       --output-dir output/ \
#       --num-gpus 8
#
# ── Single node, 2 GPUs (tp=2, 1 worker) ─────────────────────────────────
#   bash data_curation/run_curation.sh \
#       --model Qwen/Qwen3-8B \
#       --input data.jsonl \
#       --output-dir output/ \
#       --num-gpus 2 \
#       --tensor-parallel-size 2
#
# ── Multi-node (2 nodes × 8 GPUs, tp=1, 16 workers) ─────────────────────
#   # On node 0:
#   NODE_RANK=0 NUM_NODES=2 bash data_curation/run_curation.sh \
#       --model Qwen/Qwen3-4B \
#       --input data.jsonl \
#       --output-dir output/ \
#       --num-gpus 8
#
#   # On node 1:
#   NODE_RANK=1 NUM_NODES=2 bash data_curation/run_curation.sh \
#       --model Qwen/Qwen3-4B \
#       --input data.jsonl \
#       --output-dir output/ \
#       --num-gpus 8
#
# Environment variables (optional):
#   NUM_NODES   – total number of nodes (default: 1)
#   NODE_RANK   – rank of this node (default: 0)
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse --num-gpus and --tensor-parallel-size from args ─────────────────
NUM_GPUS=1
TP=1
PIPELINE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)
            NUM_GPUS="$2"; shift 2 ;;
        --tensor-parallel-size)
            TP="$2"; PIPELINE_ARGS+=("--tensor-parallel-size" "$2"); shift 2 ;;
        *)
            PIPELINE_ARGS+=("$1"); shift ;;
    esac
done

# ── Compute worker layout ────────────────────────────────────────────────
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
WORKERS_PER_NODE=$(( NUM_GPUS / TP ))
WORLD_SIZE=$(( WORKERS_PER_NODE * NUM_NODES ))

echo "=== Data Curation Launch ==="
echo "  Nodes:            ${NUM_NODES} (this node: ${NODE_RANK})"
echo "  GPUs per node:    ${NUM_GPUS}"
echo "  TP size:          ${TP}"
echo "  Workers per node: ${WORKERS_PER_NODE}"
echo "  World size:       ${WORLD_SIZE}"
echo "  Pipeline args:    ${PIPELINE_ARGS[*]}"
echo "============================"

# ── Launch workers ───────────────────────────────────────────────────────
PIDS=()
for (( LOCAL=0; LOCAL<WORKERS_PER_NODE; LOCAL++ )); do
    GLOBAL_RANK=$(( NODE_RANK * WORKERS_PER_NODE + LOCAL ))
    GPU_START=$(( LOCAL * TP ))
    GPU_END=$(( GPU_START + TP - 1 ))

    # Build CUDA_VISIBLE_DEVICES string, e.g. "0" or "2,3"
    GPUS=""
    for (( g=GPU_START; g<=GPU_END; g++ )); do
        [[ -n "$GPUS" ]] && GPUS="${GPUS},"
        GPUS="${GPUS}${g}"
    done

    echo "[Node ${NODE_RANK}] Launching worker rank=${GLOBAL_RANK} on GPU(s) ${GPUS}"

    CUDA_VISIBLE_DEVICES="${GPUS}" \
    RANK="${GLOBAL_RANK}" \
    WORLD_SIZE="${WORLD_SIZE}" \
    python "${SCRIPT_DIR}/pipeline.py" \
        --rank "${GLOBAL_RANK}" \
        --world-size "${WORLD_SIZE}" \
        "${PIPELINE_ARGS[@]}" \
        > >(sed "s/^/[rank${GLOBAL_RANK}] /") \
        2>&1 &

    PIDS+=($!)
done

# ── Wait for all workers ─────────────────────────────────────────────────
echo "Waiting for ${#PIDS[@]} workers to finish..."
FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "Worker PID ${PID} failed!"
        FAILED=1
    fi
done

if [[ $FAILED -eq 1 ]]; then
    echo "Some workers failed. Check logs above."
    exit 1
fi

echo "All workers finished successfully."
