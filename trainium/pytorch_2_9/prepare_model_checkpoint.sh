#!/usr/bin/env bash
# Download Qwen3-4B-Base and create TP=8 NxD checkpoint shards.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PREP_DIR="${PREP_DIR:-$(cd -- "${REPO_DIR}/.." && pwd)/lightning-opd-prep}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Base}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-${PREP_DIR}/model_checkpoints/qwen3-4b-base-tp8}"

mkdir -p "$(dirname -- "${MODEL_CHECKPOINT}")"
python "${SCRIPT_DIR}/convert_qwen3_checkpoint.py" \
    --model-id "${MODEL_ID}" \
    --output-dir "${MODEL_CHECKPOINT}" \
    --tp-size 8

test -f "${MODEL_CHECKPOINT}/pretrained_weight/model/dp_rank_00_tp_rank_00_pp_rank_00.pt"
echo "MODEL_CHECKPOINT_OK: ${MODEL_CHECKPOINT}"
