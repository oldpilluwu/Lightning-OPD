#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Trainium port of data_curation/run_curation.sh.
#
# Launches one independent vLLM-on-Neuron worker per group of NeuronCores.
# Device isolation uses NEURON_RT_VISIBLE_CORES (the Neuron equivalent of
# CUDA_VISIBLE_DEVICES). A trn2.3xlarge exposes 4 logical NeuronCores (0-3)
# under the default LNC=2 config; a trn1.32xlarge exposes 32 (0-31).
#
#   bash trainium/run_curation_neuron.sh \
#       --model Qwen/Qwen3-8B \
#       --input data.jsonl \
#       --output-dir output/ \
#       --num-cores 4 \
#       --tensor-parallel-size 4      # -> 1 worker x 4 cores (trn2.3xlarge)
#
# Environment variables (optional):
#   NUM_NODES / NODE_RANK - multi-instance sharding (default: 1 / 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_CORES=4
TP=1
PIPELINE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-cores)
            NUM_CORES="$2"; shift 2 ;;
        --tensor-parallel-size)
            TP="$2"; PIPELINE_ARGS+=("--tensor-parallel-size" "$2"); shift 2 ;;
        *)
            PIPELINE_ARGS+=("$1"); shift ;;
    esac
done

NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
if (( NUM_CORES % TP != 0 )); then
    echo "ERROR: --num-cores (${NUM_CORES}) must be divisible by --tensor-parallel-size (${TP})." >&2
    echo "       Otherwise workers-per-node floors and some NeuronCores go unused / mis-sharded." >&2
    exit 1
fi
WORKERS_PER_NODE=$(( NUM_CORES / TP ))
WORLD_SIZE=$(( WORKERS_PER_NODE * NUM_NODES ))

echo "=== Data Curation Launch (Neuron) ==="
echo "  Nodes:             ${NUM_NODES} (this node: ${NODE_RANK})"
echo "  NeuronCores/node:  ${NUM_CORES}"
echo "  TP size:           ${TP}"
echo "  Workers per node:  ${WORKERS_PER_NODE}"
echo "  World size:        ${WORLD_SIZE}"
echo "  Pipeline args:     ${PIPELINE_ARGS[*]}"
echo "====================================="

PIDS=()
for (( LOCAL=0; LOCAL<WORKERS_PER_NODE; LOCAL++ )); do
    GLOBAL_RANK=$(( NODE_RANK * WORKERS_PER_NODE + LOCAL ))
    CORE_START=$(( LOCAL * TP ))
    CORE_END=$(( CORE_START + TP - 1 ))

    echo "[Node ${NODE_RANK}] Launching worker rank=${GLOBAL_RANK} on NeuronCores ${CORE_START}-${CORE_END}"

    NEURON_RT_VISIBLE_CORES="${CORE_START}-${CORE_END}" \
    RANK="${GLOBAL_RANK}" \
    WORLD_SIZE="${WORLD_SIZE}" \
    python "${SCRIPT_DIR}/pipeline_neuron.py" \
        --rank "${GLOBAL_RANK}" \
        --world-size "${WORLD_SIZE}" \
        "${PIPELINE_ARGS[@]}" \
        > >(sed "s/^/[rank${GLOBAL_RANK}] /") \
        2>&1 &

    PIDS+=($!)
done

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
