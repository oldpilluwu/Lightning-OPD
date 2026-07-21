#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge parquet shards produced by curate_qwen3_8b_2x8x3090_vast.sh.

Example:
    python scripts/merge_vast_qwen3_8b_shards.py \
        --inputs \
            data/sft_data/openthoughts3_300000_qwen3-8b_node0-of-2.parquet \
            data/sft_data/openthoughts3_300000_qwen3-8b_node1-of-2.parquet \
        --output data/sft_data/openthoughts3_300000_qwen3-8b.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Vast.ai node parquet shards.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Parquet shards from each Vast node.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Final merged parquet path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.inputs]

    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input shard(s): {missing}")

    frames = []
    for path in input_paths:
        df = pd.read_parquet(path)
        print(f"{path}: {len(df)} rows")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output, index=False)
    print(f"Merged {len(merged)} rows -> {output}")


if __name__ == "__main__":
    main()
