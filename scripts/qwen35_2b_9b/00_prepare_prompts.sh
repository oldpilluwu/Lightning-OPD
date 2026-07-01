#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

mkdir -p "$(dirname "${SFT_PROMPTS}")"

python scripts/prepare_sft_prompts.py \
    --hf-dataset open-thoughts/OpenThoughts3-1.2M \
    --output "${SFT_PROMPTS}" \
    --num-samples "${SFT_NUM_SAMPLES}"

huggingface-cli download zhuzilin/dapo-math-17k \
    --repo-type dataset \
    --include "*.jsonl" \
    --local-dir "$(dirname "${OPD_PROMPTS}")"

echo "SFT prompts: ${SFT_PROMPTS}"
echo "OPD prompts: ${OPD_PROMPTS}"
