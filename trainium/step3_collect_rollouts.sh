#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Step 3 (Trainium): collect student rollouts on OPD prompts (DAPO-Math-17k)
# with the SFT model running on Neuron. Mirrors scripts/collect_rollouts.sh.
#
# Required:
#   SFT_CHECKPOINT - path to the SFT model (consolidated HF format)
#   OPD_PROMPTS    - OPD prompt dataset (.jsonl or .parquet)
#   OUTPUT_DIR     - output directory
# Optional:
#   NUM_CORES      - NeuronCores to use (default: 32)
#   TP_SIZE        - NeuronCores per vLLM worker (default: 8)

set -euo pipefail

: "${SFT_CHECKPOINT:?Set SFT_CHECKPOINT to the SFT model path}"
: "${OPD_PROMPTS:?Set OPD_PROMPTS to the OPD prompt dataset path}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for collected rollout data}"

SFT_CHECKPOINT="$(cd "$(dirname "${SFT_CHECKPOINT}")" && pwd)/$(basename "${SFT_CHECKPOINT}")"
OPD_PROMPTS="$(cd "$(dirname "${OPD_PROMPTS}")" && pwd)/$(basename "${OPD_PROMPTS}")"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"

NUM_CORES="${NUM_CORES:-32}"
TP_SIZE="${TP_SIZE:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_curation_neuron.sh" \
    --model "${SFT_CHECKPOINT}" \
    --input "${OPD_PROMPTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-cores "${NUM_CORES}" \
    --tensor-parallel-size "${TP_SIZE}" \
    "$@"
