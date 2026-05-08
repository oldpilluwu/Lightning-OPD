# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert HuggingFace OpenThoughts3-1.2M dataset to a prompt-only JSONL file
for SFT data generation (Step 1).

Extracts the prompt (user messages) from each sample and writes to JSONL.
Optionally samples a subset (default 300K) to reduce compute cost.

Usage:
    python scripts/prepare_sft_prompts.py \
        --output data/prompts/openthoughts3_300k.jsonl \
        --num-samples 300000

    # Use a local parquet file instead of downloading from HF
    python scripts/prepare_sft_prompts.py \
        --input data/prompts/local.parquet \
        --output data/prompts/openthoughts3_300k.jsonl
"""

import argparse
import json
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract prompts from OpenThoughts3-1.2M for SFT data generation."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to a local parquet/jsonl file. If not set, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--hf-dataset", type=str, default="open-thoughts/OpenThoughts3-1.2M",
        help="HuggingFace dataset name (default: open-thoughts/OpenThoughts3-1.2M).",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=300000,
        help="Number of samples to keep (default: 300000). Set to 0 for all.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42).",
    )
    return parser.parse_args()


def extract_prompt(sample):
    """Extract the prompt (non-assistant messages) from a sample.

    Supports two common formats:
    1. {"conversations": [{"from": "human", "value": ...}, ...]}  (sharegpt)
    2. {"prompt": [{"role": "user", "content": ...}, ...]}  (chat messages)
    """
    if "conversations" in sample:
        messages = []
        for turn in sample["conversations"]:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            if role in ("human", "user"):
                messages.append({"role": "user", "content": content})
            elif role == "system":
                messages.append({"role": "system", "content": content})
        if messages:
            return {"prompt": messages}

    if "prompt" in sample:
        if isinstance(sample["prompt"], list):
            return {"prompt": sample["prompt"]}
        elif isinstance(sample["prompt"], str):
            return {"prompt": [{"role": "user", "content": sample["prompt"]}]}

    if "messages" in sample:
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in sample["messages"]
            if m["role"] != "assistant"
        ]
        if messages:
            return {"prompt": messages}

    return None


def load_dataset_from_hf(dataset_name):
    """Load dataset from HuggingFace."""
    from datasets import load_dataset
    print(f"Loading dataset from HuggingFace: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    return ds


def load_dataset_from_file(path):
    """Load dataset from local file (parquet or jsonl)."""
    import pandas as pd
    print(f"Loading dataset from local file: {path}")
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        return df.to_dict("records")
    elif path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unsupported format: {path}")


def main():
    args = parse_args()
    random.seed(args.seed)

    # Load dataset
    if args.input:
        samples = load_dataset_from_file(args.input)
    else:
        samples = load_dataset_from_hf(args.hf_dataset)

    print(f"Total samples: {len(samples)}")

    # Sample subset
    if args.num_samples > 0 and args.num_samples < len(samples):
        indices = random.sample(range(len(samples)), args.num_samples)
        indices.sort()
        samples = [samples[i] for i in indices]
        print(f"Sampled {args.num_samples} samples")

    # Extract prompts
    from tqdm import tqdm
    written = 0
    skipped = 0
    with open(args.output, "w") as f:
        for sample in tqdm(samples, desc="Extracting prompts"):
            prompt_item = extract_prompt(sample)
            if prompt_item and len(prompt_item["prompt"]) > 0:
                f.write(json.dumps(prompt_item) + "\n")
                written += 1
            else:
                skipped += 1

    print(f"Written: {written}, Skipped: {skipped}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
