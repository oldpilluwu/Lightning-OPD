#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Launch native PyTorch workers as independent replicas or one DTensor TP group.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NUM_CORES=1
TP_SIZE=1
PIPELINE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-cores) NUM_CORES="$2"; shift 2 ;;
        --tensor-parallel-size) TP_SIZE="$2"; shift 2 ;;
        *) PIPELINE_ARGS+=("$1"); shift ;;
    esac
done

NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
WORLD_SIZE=$((NUM_CORES * NUM_NODES))
cd "${REPO_ROOT}"

if [[ "${TP_SIZE}" -gt 1 ]]; then
    [[ "${NUM_NODES}" == "1" ]] || { echo "ERROR: native TP launcher currently supports one node." >&2; exit 1; }
    [[ "${NUM_CORES}" == "${TP_SIZE}" ]] || {
        echo "ERROR: set NUM_CORES equal to TP_SIZE for one native TP replica." >&2; exit 1;
    }
fi

echo "=== Native PyTorch data curation ==="
echo "mode:        Transformers + TorchNeuron (no vLLM/XLA/NxD)"
echo "node:        ${NODE_RANK}/${NUM_NODES}"
echo "workers:     ${NUM_CORES}"
echo "TP size:     ${TP_SIZE}"
echo "world size:  ${WORLD_SIZE}"

if [[ "${TP_SIZE}" -gt 1 ]]; then
    export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:-} --model-type=transformer"
    exec torchrun \
        --standalone \
        --nproc-per-node "${NUM_CORES}" \
        -m trainium.sft_data_generation_native.pipeline \
        --tensor-parallel-size "${TP_SIZE}" \
        "${PIPELINE_ARGS[@]}"
fi

PIDS=()
for ((LOCAL_RANK=0; LOCAL_RANK<NUM_CORES; LOCAL_RANK++)); do
    GLOBAL_RANK=$((NODE_RANK * NUM_CORES + LOCAL_RANK))
    echo "launching rank ${GLOBAL_RANK} on logical NeuronCore ${LOCAL_RANK}"
    NEURON_RT_VISIBLE_CORES="${LOCAL_RANK}" \
    RANK="${GLOBAL_RANK}" \
    WORLD_SIZE="${WORLD_SIZE}" \
    python -m trainium.sft_data_generation_native.pipeline \
        --rank "${GLOBAL_RANK}" \
        --world-size "${WORLD_SIZE}" \
        "${PIPELINE_ARGS[@]}" \
        > >(sed "s/^/[rank${GLOBAL_RANK}] /") 2>&1 &
    PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
    wait "${PID}" || FAILED=1
done
if [[ "${FAILED}" == "1" ]]; then
    echo "ERROR: one or more native curation workers failed." >&2
    exit 1
fi
echo "All native curation workers finished."
