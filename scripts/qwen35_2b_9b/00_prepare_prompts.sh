#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

mkdir -p "$(dirname "${SFT_PROMPTS}")"

python scripts/prepare_sft_prompts.py \
    --hf-dataset open-thoughts/OpenThoughts3-1.2M \
    --output "${SFT_PROMPTS}" \
    --num-samples "${SFT_NUM_SAMPLES}"

if command -v hf >/dev/null 2>&1; then
    hf download zhuzilin/dapo-math-17k \
        --repo-type dataset \
        --include "*.jsonl" \
        --local-dir "$(dirname "${OPD_PROMPTS}")"
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download zhuzilin/dapo-math-17k \
        --repo-type dataset \
        --include "*.jsonl" \
        --local-dir "$(dirname "${OPD_PROMPTS}")"
else
    echo "Missing Hugging Face CLI. Install with: python -m pip install 'huggingface_hub[cli]'" >&2
    exit 1
fi

echo "SFT prompts: ${SFT_PROMPTS}"
echo "OPD prompts: ${OPD_PROMPTS}"
